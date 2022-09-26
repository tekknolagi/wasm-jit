# WASM JIT workbench

This repo is for artifacts related to run-time code generation for
WebAssembly programs and components.

## Background

### What's the big idea?

Just-in-time (JIT) code generation is an important tactic when
implementing a programming language.  Generating code at run-time allows
a program to specialize itself against the specific data it is run
against.  For a program that implements a programming language, that
specialization is with respect to the program being run, and possibly
with respect to the data that program uses.

The way this typically works is that the program generates bytes for the
instruction set of the machine it's running on, and then transfers
control to those instructions.

Usually the program has to put its generated code in memory that is
specially marked as executable.  However, this capability is missing in
WebAssembly.  How, then, to do just-in-time compilation in WebAssembly?

### WebAssembly as a Harvard architecture

In a von Neumman machine, like the ones that you are probably reading
this on, code and data share an address space.  There's only one kind of
pointer, and it can point to anything: the bytes that implement the
`sin` function, the number `42`, the characters in `"biscuits"`, or
anything at all.  WebAssembly is different in that its code is not
addressable at run-time.  Functions in a WebAssembly module are numbered
sequentially from 0, and the WebAssembly `call` instruction takes the
callee as an immediate parameter.

So, to add code to a WebAssembly program, somehow you'd have to augment
the program with more functions.  Let's assume we will make that
possible somehow -- that your WebAssembly module that had N functions
will now have N+1 functions, and with function N being the new one your
program generated.  How would we call it?  Given that the `call`
instructions hard-code the callee, the existing functions 0 to N-1 won't
call it.

Here the answer is `call_indirect`.  A bit of a reminder, this
instruction take the callee as an operand, not an immediate parameter,
allowing it to choose the callee function at run-time.  The callee
operand is an index into a *table* of functions.  Conventionally, table
0 is called the *indirect call table* as it contains an entry for each
function which might ever be the target of an indirect call.

With this in mind, our problem has two parts, then: (1) how to augment a
WebAssembly module with a new function, and (2) how to get the original
module to call the new code.

### Late linking of auxiliary WebAssembly modules

The key idea here is that to add code, the main program should generate
a new WebAssembly module containing that code.  Then we run a linking
phase to actually bring that new code to life and make it available.

System linkers like `ld` typically require a complete set of symbols and
relocations to resolve inter-archive references.  However when
performing a late link of JIT-generated code, we can take a short-cut:
the main program can embed memory addresses directly into the code it
generates.  Therefore the generated module will import memory from the
main module.  All references from the generated code to the main module
can be directly embedded in this way.

The generated module will also import the indirect function table from
the main module.  (We will ensure that the main module exports its
memory and indirect function table via the toolchain.)  When the main
module makes the generated module, it also embeds a special `patch`
function in the generated module.  This function will add the new
functions to the main module's indirect function table, and perform any
relocations onto the main module's memory.  All references from the main
module to generated functions are installed via the `patch` function.

We plan on two implementations of late linking, but both share the
fundamental mechanism of a generated WebAssembly module with a `patch`
function.

#### Dynamic linking via the run-time

One implementation of a linker is for the main module to cause the
run-time to dynamically instantiate a new WebAssembly module.  The
run-time would provide the memory and indirect function table from the
main module as imports when instantiating the generated module.

The advantage of dynamic linking is that it can update a live
WebAssembly module without any need for re-instantiation or special
run-time checkpointing support.

#### Static linking via Wizer

Another idea is to build on
[Wizer](https://github.com/bytecodealliance/wizer)'s ability to take a
snapshot of a WebAssembly module.  We will extend Wizer to also be able
to augment a module with new code.  In this role, Wizer is effectively a
late linker, linking in a new archive to an existing object.

Wizer already needs the ability to instantiate a WebAssembly module and
to run its code.  Causing Wizer to ask the module if it has any
generated auxiliary module that should be instantiated, patched, and
incorporated into the main module should not be a huge deal.

### Late linking appears to be async codegen

From the perspective of a main program, WebAssembly JIT code generation
via late linking appears the same as aynchronous code generation.

For example, take the C program:

```c
struct Value;
struct Func {
  struct Expr *body;
  void *jitCode;
};

void recordJitCandidate(struct Func *func);
uint8_t* flushJitCode(); // Call to actually generate JIT code.

struct Value* interpretCall(struct Expr *body,
                            struct Value *arg);

struct Value* call(struct Func *func,
                   struct Value* val) {
  if (func->jitCode) {
    struct Value* (*f)(struct Value*) = func->jitCode;
    return f(val);
  } else {
    recordJitCandidate(func);
    return interpretCall(func->body, val);
  }
}
```

Here the C program allows for the possibility of JIT code generation:
there is a slot in a `Func` instance to fill in with a code pointer.  If
this program generates code for a given `Func`, it won't be able to fill
in the pointer -- it can't add new code to the image.  But, it could
tell Wizer to do so, and Wizer could snapshot the program, link in the
new function, and patch `&func->jitCode`.  From the program's
perspective, it's as if the code becomes available asynchronously.

## Usage

In this repository we have `interp.cc` which implements a little Scheme
interpreter and JIT compiler.  You compile it like this:

```
$ /opt/wasi-sdk/bin/clang++ -O2 -Wall \
   -mexec-model=reactor \
   -Wl,--growable-table \
   -Wl,--export-table \
   -DLIBRARY=1 \
   -fno-exceptions \
   interp.cc -o interplib.wasm
```

We compile with the WASI SDK.  I have version 14.

The `-mexec-model=reactor` means that this WASI module isn't just a
run-once thing, after which its state is torn down; rather it's a
multiple-entry component.

The two `-Wl,` options tell the linker to export the indirect function
table, and to allow the indirect function table to be augmented by the
JIT module.

The `-DLIBRARY=1` is used by `interp.cc`; you can actually run and debug
it natively but that's just for development.  We're instead compiling to
wasm and running with a WASI environment (wasmtime).

The `-fno-exceptions` is because WASI doesn't support exceptions
currently.  Also we don't need them.

To run the module, we have a little harness that uses the [the Python
interface to
wasmtime](https://github.com/bytecodealliance/wasmtime-py/).  That
harness includes a little factorial function as an example program.

```
$ python3 interp.py
Parsing: 
(letrec ((fib (lambda (n)
               (if (< n 2)
                   1
                   (+ (fib (- n 1))
                      (fib (- n 2)))))))
  (fib 30))

Parse result: 0x11eb0
Calling eval(0x11eb0) 5 times
result: 1346269
result: 1346269
result: 1346269
result: 1346269
result: 1346269
Calling eval(0x11eb0) 5 times took 2.3992819786071777s.
Calling jitModule()
jitModule result: <wasmtime._module.Module object at 0x7f2bef0821c0>
Instantiating and patching in JIT module
Calling eval(0x11eb0) 5 times
result: 1346269
result: 1346269
result: 1346269
result: 1346269
result: 1346269
Calling eval(0x11eb0) 5 times took 1.3382737636566162s.
```

After calling eval the first five times, we ask the module to generate
any JIT code that it would like.  Then we instantiate that JIT code,
patch it into memory, and run the eval test 5 more times.  This time the
interpreter sees that there is JIT code and runs it instead.  The result
is the same but we get it faster.

The next task here is to implement linking via Wizer, but already this
is a good proof that you can indeed JIT with WASM :)
