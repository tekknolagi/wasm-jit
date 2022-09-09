#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cassert>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#ifdef LIBRARY
#define EXPORT(name) __attribute__((export_name(name)))
void throw_error() __attribute__((noreturn))
__attribute__((import_module("env")))
__attribute__((import_name("throw_error")));
#else
#define EXPORT(name) static
static void throw_error() __attribute__((noreturn));
static void throw_error() { exit(1); }
#endif

__attribute__((noreturn)) static void signal_error(const char* message,
                                                   const char* what) {
  if (what) {
    fprintf(stderr, "error: %s: %s\n", message, what);
  } else {
    fprintf(stderr, "error: %s\n", message);
  }
  throw_error();
}

class Expr {
 public:
  enum class Kind { Func, LetRec, Var, Prim, Literal, Call, If };

  const Kind kind;

  virtual std::string toString() const = 0;

  Expr() = delete;
  virtual ~Expr() {}

 protected:
  Expr(Kind kind) : kind(kind) {}
};

template <typename I>
std::string n2hexstr(I w, size_t hex_len = sizeof(I) << 1) {
  static const char* digits = "0123456789abcdef";
  std::string rc(hex_len, '0');
  for (size_t i = 0, j = (hex_len - 1) * 4; i < hex_len; ++i, j -= 4) {
    rc[i] = digits[(w >> j) & 0x0f];
  }
  return rc;
}

class Func : public Expr {
 public:
  const std::unique_ptr<Expr> body;
  void* jitCode;

  // FIXME: We need to be able to get to the body from JIT code.  Does this mean
  // we shouldn't be using unique_ptr ?
  static size_t offsetOfBody() { return sizeof(Expr); }
  static size_t offsetOfJitCode() { return offsetOfBody() + sizeof(body); }

  std::string toString() const {
    return "(func " + n2hexstr(reinterpret_cast<uintptr_t>(jitCode)) + " " +
           body->toString() + ")";
  }

  explicit Func(Expr* body) : Expr(Kind::Func), body(body), jitCode(nullptr) {}
};

// letrec is a bit of a misnomer; it is actually normal let
class LetRec : public Expr {
 public:
  const std::vector<std::unique_ptr<Expr>> args;
  const std::unique_ptr<Expr> body;

  std::string toString() const {
    std::string result = "(letrec";
    for (size_t i = 0; i < args.size(); i++) {
      result += " " + args[i]->toString();
    }
    result += " " + body->toString() + ")";
    return result;
  }

  LetRec(std::vector<std::unique_ptr<Expr>>&& args, Expr* body)
      : Expr(Kind::LetRec), args(std::move(args)), body(body) {}
};

class Var : public Expr {
 public:
  uint32_t depth;

  std::string toString() const { return "(var " + std::to_string(depth) + ")"; }

  explicit Var(uint32_t depth) : Expr(Kind::Var), depth(depth){};
};

class Prim : public Expr {
 public:
  enum class Op { Eq, LessThan, Sub, Add, Mul };

  const Op op;
  const std::unique_ptr<Expr> lhs;
  const std::unique_ptr<Expr> rhs;

  static std::string opToString(Op op) {
    switch (op) {
      case Op::Eq:
        return "=";
      case Op::LessThan:
        return "<";
      case Op::Sub:
        return "-";
      case Op::Add:
        return "+";
      case Op::Mul:
        return "*";
    }
    signal_error("unhandled op", nullptr);
  }

  std::string toString() const {
    return "(" + opToString(op) + " " + lhs->toString() + " " +
           rhs->toString() + ")";
  }

  Prim(Op op, Expr* lhs, Expr* rhs)
      : Expr(Kind::Prim), op(op), lhs(lhs), rhs(rhs){};
};

class Literal : public Expr {
 public:
  const int32_t val;

  std::string toString() const { return "(lit " + std::to_string(val) + ")"; }

  Literal(int32_t val) : Expr(Kind::Literal), val(val){};
};

class Call : public Expr {
 public:
  const std::unique_ptr<Expr> func;
  const std::vector<std::unique_ptr<Expr>> args;

  std::string toString() const {
    std::string result = "(" + func->toString();
    for (const std::unique_ptr<Expr>& arg : args) {
      result += " " + arg->toString();
    }
    result += ")";
    return result;
  }

  Call(Expr* func, std::vector<std::unique_ptr<Expr>>&& args)
      : Expr(Kind::Call), func(func), args(std::move(args)){};
};

class If : public Expr {
 public:
  const std::unique_ptr<Expr> test;
  const std::unique_ptr<Expr> consequent;
  const std::unique_ptr<Expr> alternate;

  std::string toString() const {
    return "(if " + test->toString() + " " + consequent->toString() + " " +
           alternate->toString() + ")";
  }

  If(Expr* test, Expr* consequent, Expr* alternate)
      : Expr(Kind::If),
        test(test),
        consequent(consequent),
        alternate(alternate){};
};

class Parser {
  std::vector<std::string> boundVars;

  void pushBound(std::string&& id) {
    boundVars.push_back(id);
          fprintf(stderr, "pushed %s\n", boundVars.at(boundVars.size() - 1).c_str());
  }
  void popBound() {
          fprintf(stderr, "popped %s\n", boundVars.at(boundVars.size() - 1).c_str());
    boundVars.pop_back();
  }
  uint32_t lookupBound(const std::string& id) {
      fprintf(stderr, "... looking up %s ... ", id.c_str());
    for (size_t i = 0; i < boundVars.size(); i++) {
      if (boundVars[boundVars.size() - i - 1] == id) {
        fprintf(stderr, "found at %lu\n", i);
        return i;
      }
    }
    signal_error("unbound identifier", id.c_str());
    return -1;
  }

  const char* buf;
  size_t pos;
  size_t len;

  static bool isAlphabetic(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
  }
  static bool isNumeric(char c) { return '0' <= c && c <= '9'; }
  static bool isAlphaNumeric(char c) { return isAlphabetic(c) || isNumeric(c); }
  static bool isWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  void error(const char* message) {
    signal_error(message, eof() ? buf : buf + pos);
  }
  bool eof() const { return pos == len; }
  char peek() {
    if (eof()) {
      return '\0';
    }
    return buf[pos];
  }
  void advance() {
    if (!eof()) {
      pos++;
    }
  }
  char next() {
    char ret = peek();
    advance();
    return ret;
  }
  bool matchChar(char c) {
    if (eof() || peek() != c) {
      return false;
    }
    advance();
    return true;
  }
  void skipWhitespace() {
    while (!eof() && isWhitespace(peek())) {
      advance();
    }
  }
  bool startsIdentifier() { return !eof() && isAlphabetic(peek()); }
  bool continuesIdentifier() { return !eof() && isAlphaNumeric(peek()); }
  bool matchIdentifier(const char* literal) {
    size_t match_len = std::strlen(literal);
    if (match_len < (len - pos)) {
      return false;
    }
    if (strncmp(buf + pos, literal, match_len) != 0) {
      return false;
    }
    if ((len - pos) < match_len && isAlphaNumeric(buf[pos + match_len])) {
      return false;
    }
    pos += match_len;
    return true;
  }
  std::string takeIdentifier() {
    size_t start = pos;
    while (continuesIdentifier()) {
      advance();
    }
    size_t end = pos;
    return std::string(buf + start, end - start);
  }
  bool matchKeyword(const char* kw) {
    size_t kwlen = std::strlen(kw);
    size_t remaining = len - pos;
    if (remaining < kwlen) {
      return false;
    }
    if (strncmp(buf + pos, kw, kwlen) != 0) {
      return false;
    }
    pos += kwlen;
    if (!continuesIdentifier()) {
      return true;
    }
    pos -= kwlen;
    return false;
    if ((len - pos) < kwlen && isalnum(buf[pos + kwlen])) {
      return 0;
    }
    pos += kwlen;
    return 1;
  }
  Expr* parsePrim(Prim::Op op) {
    Expr* lhs = parseOne();
    Expr* rhs = parseOne();
    return new Prim(op, lhs, rhs);
  }
  int32_t parseInt32() {
    uint64_t ret = 0;
    while (!eof() && isNumeric(peek())) {
      ret *= 10;
      ret += next() - '0';
      if (ret > 0x7fffffff) {
        error("integer too long");
      }
    }
    if (!eof() && !isWhitespace(peek()) && peek() != ')') {
      error("unexpected integer suffix");
    }
    return ret;
  }
  Expr* parseOne() {
    skipWhitespace();
    if (eof()) {
      error("unexpected end of input");
    }
    if (matchChar('(')) {
      skipWhitespace();

      Expr* ret;
      if (matchKeyword("lambda")) {
        skipWhitespace();
        if (!matchChar('(')) {
          error("expected open paren after lambda");
        }
        skipWhitespace();
        int to_pop = 0;
        while (!matchChar(')')) {
          pushBound(takeIdentifier());
          skipWhitespace();
          to_pop++;
        }
        Expr* body = parseOne();
        while (to_pop--) {
          popBound();
        }
        ret = new Func(body);
      } else if (matchKeyword("letrec")) {
        skipWhitespace();
        if (!matchChar('(')) {
          error("expected open paren after letrec");
        }
        int to_pop = 0;
        std::vector<std::unique_ptr<Expr>> args;
        while (!matchChar(')')) {
          skipWhitespace();
          if (!matchChar('(')) {
            error("expected open paren in letrec binding");
          }
          skipWhitespace();
          if (!startsIdentifier()) {
            error("expected an identifier for letrec");
          }
          pushBound(takeIdentifier());
          skipWhitespace();
          args.emplace_back(parseOne());
          skipWhitespace();
          if (!matchChar(')')) {
            error("expected close paren in letrec binding");
          }
          skipWhitespace();
          to_pop++;
        }
        skipWhitespace();
        Expr* body = parseOne();
        while (to_pop--) {
          popBound();
        }
        ret = new LetRec(std::move(args), body);
      } else if (matchKeyword("+")) {
        ret = parsePrim(Prim::Op::Add);
      } else if (matchKeyword("-")) {
        ret = parsePrim(Prim::Op::Sub);
      } else if (matchKeyword("<")) {
        ret = parsePrim(Prim::Op::LessThan);
      } else if (matchKeyword("eq?")) {
        ret = parsePrim(Prim::Op::Eq);
      } else if (matchKeyword("*")) {
        ret = parsePrim(Prim::Op::Mul);
      } else if (matchKeyword("if")) {
        Expr* test = parseOne();
        Expr* consequent = parseOne();
        Expr* alternate = parseOne();
        ret = new If(test, consequent, alternate);
      } else {
        // Otherwise it's a call.
        Expr* func = parseOne();
        std::vector<std::unique_ptr<Expr>> args;
        while (!matchChar(')')) {
          args.emplace_back(parseOne());
          skipWhitespace();
        }
        return new Call(func, std::move(args));
      }
      skipWhitespace();
      if (!matchChar(')')) {
        error("expected close parenthesis");
      }
      return ret;
    }
    if (startsIdentifier()) {
      return new Var(lookupBound(takeIdentifier()));
    }
    if (isNumeric(peek())) {
      return new Literal(parseInt32());
    }
    error("unexpected input");
    return nullptr;
  }

 public:
  explicit Parser(const char* buf) : buf(buf), pos(0), len(strlen(buf)) {}

  Expr* parse() {
    Expr* e = parseOne();
    skipWhitespace();
    if (!eof()) {
      error("expected end of input after expression");
    }
    return e;
  }
};

EXPORT("parse")
Expr* parse(const char* str) { return Parser(str).parse(); }

#define FOR_EACH_HEAP_OBJECT_KIND(M)                                           \
  M(env, Env)                                                                  \
  M(closure, Closure)

#define DECLARE_CLASS(name, Name) class Name;
FOR_EACH_HEAP_OBJECT_KIND(DECLARE_CLASS)
#undef DECLARE_CLASS

class Heap;

class HeapObject {
 public:
  // Any other kind value indicates a forwarded object.
  enum class Kind : uintptr_t {
#define DECLARE_KIND(name, Name) Name,
    FOR_EACH_HEAP_OBJECT_KIND(DECLARE_KIND)
#undef DECLARE_KIND
  };

  static const uintptr_t kNotForwardedBit = 1;
  static const uintptr_t kNotForwardedBits = 1;
  static const uintptr_t kNotForwardedBitMask = (1 << kNotForwardedBits) - 1;

 protected:
  uintptr_t tag;

  HeapObject(Kind kind)
      : tag((static_cast<uintptr_t>(kind) << kNotForwardedBits) |
            kNotForwardedBit) {}

 public:
  static size_t offsetOfTag() { return 0; }

  bool isForwarded() const { return (tag & kNotForwardedBit) == 0; }
  HeapObject* forwarded() const { return reinterpret_cast<HeapObject*>(tag); }
  void forward(HeapObject* new_loc) {
    tag = reinterpret_cast<uintptr_t>(new_loc);
  }

  Kind kind() const { return static_cast<Kind>(tag >> 1); }

#define DEFINE_METHODS(name, Name)                                             \
  bool is##Name() const { return kind() == Kind::Name; }                       \
  Name* as##Name() { return reinterpret_cast<Name*>(this); }
  FOR_EACH_HEAP_OBJECT_KIND(DEFINE_METHODS)
#undef DEFINE_METHODS

  const char* kindName() const {
    switch (kind()) {
#define RETURN_KIND_NAME(name, Name)                                           \
  case Kind::Name:                                                             \
    return #name;
      FOR_EACH_HEAP_OBJECT_KIND(RETURN_KIND_NAME)
#undef RETURN_KIND_NAME
      default:
        signal_error("unexpected heap object kind", nullptr);
        return nullptr;
    }
  }
  inline void* operator new(size_t nbytes, Heap& heap);
};

class Value;

class Heap {
  uintptr_t hp;
  uintptr_t limit;
  uintptr_t base;
  size_t size;
  long count;
  char* mem;

  std::vector<Value> roots;

  static const uintptr_t kAlignment = 8;

  static uintptr_t alignUp(uintptr_t val) {
    return (val + kAlignment - 1) & ~(kAlignment - 1);
  }

  void flip() {
    uintptr_t split = base + (size >> 1);
    if (hp <= split) {
      hp = split;
      limit = base + size;
    } else {
      hp = base;
      limit = split;
    }
    count++;
  }

  HeapObject* copy(HeapObject* obj);
  size_t scan(HeapObject* obj);

  void visitRoots();

  void collect() {
    flip();
    uintptr_t grey = hp;
    visitRoots();
    while (grey < hp) {
      grey += alignUp(scan(reinterpret_cast<HeapObject*>(grey)));
    }
  }

 public:
  explicit Heap(size_t heap_size) {
    mem = new char[alignUp(heap_size)];
    if (!mem) {
      signal_error("malloc failed", nullptr);
    }

    hp = base = reinterpret_cast<uintptr_t>(mem);
    size = heap_size;
    count = -1;
    flip();
  }
  ~Heap() { delete[] mem; }

  static size_t pushRoot(Heap* heap, Value v);
  static Value getRoot(Heap* heap, size_t idx);
  static void setRoot(Heap* heap, size_t idx, Value v);
  static void popRoot(Heap* heap);

  template <typename T>
  void visit(T** loc) {
    HeapObject* obj = *loc;
    if (obj != nullptr) {
      *loc = static_cast<T*>(obj->isForwarded() ? obj->forwarded() : copy(obj));
    }
  }

  inline HeapObject* allocate(size_t size) {
    while (1) {
      uintptr_t addr = hp;
      uintptr_t new_hp = alignUp(addr + size);
      if (limit < new_hp) {
        collect();
        if (limit - hp < size) {
          signal_error("ran out of space", nullptr);
        }
        continue;
      }
      hp = new_hp;
      return reinterpret_cast<HeapObject*>(addr);
    }
  }
};

inline void* HeapObject::operator new(size_t bytes, Heap& heap) {
  return heap.allocate(bytes);
}

class Value {
  uintptr_t payload;

 public:
  static const uintptr_t kHeapObjectTag = 0;
  static const uintptr_t kSmiTag = 1;
  static const uintptr_t kTagBits = 1;
  static const uintptr_t kTagMask = (1 << kTagBits) - 1;

  explicit Value(HeapObject* obj) : payload(reinterpret_cast<uintptr_t>(obj)) {}
  explicit Value(intptr_t val)
      : payload((static_cast<uintptr_t>(val) << kTagBits) | kSmiTag) {}

  bool isSmi() const { return (payload & kTagBits) == kSmiTag; }
  bool isHeapObject() const { return (payload & kTagMask) == kHeapObjectTag; }
  intptr_t getSmi() const { return static_cast<intptr_t>(payload) >> kTagBits; }
  HeapObject* getHeapObject() {
    return reinterpret_cast<HeapObject*>(payload & ~kHeapObjectTag);
  }
  uintptr_t bits() { return payload; }

  const char* kindName() {
    return isSmi() ? "small integer" : getHeapObject()->kindName();
  }

#define DEFINE_METHODS(name, Name)                                             \
  bool is##Name() { return isHeapObject() && getHeapObject()->is##Name(); }    \
  Name* as##Name() { return getHeapObject()->as##Name(); }
  FOR_EACH_HEAP_OBJECT_KIND(DEFINE_METHODS)
#undef DEFINE_METHODS

  void visitFields(Heap& heap) {
    if (isHeapObject()) {
      heap.visit(reinterpret_cast<HeapObject**>(&payload));
    }
  }
};

size_t Heap::pushRoot(Heap* heap, Value v) {
  size_t ret = heap->roots.size();
  heap->roots.push_back(v);
  return ret;
}
Value Heap::getRoot(Heap* heap, size_t idx) { return heap->roots[idx]; }
void Heap::setRoot(Heap* heap, size_t idx, Value v) { heap->roots[idx] = v; }
void Heap::popRoot(Heap* heap) { heap->roots.pop_back(); }

template <typename T>
class Rooted {
  Heap& heap;
  size_t idx;

 public:
  Rooted(Heap& heap, T* obj)
      : heap(heap), idx(Heap::pushRoot(&heap, Value(obj))) {}
  ~Rooted() { Heap::popRoot(&heap); }

  T* get() const {
    return static_cast<T*>(Heap::getRoot(&heap, idx).getHeapObject());
  }
  void set(T* obj) { Heap::setRoot(&heap, idx, Value(obj)); }
};

template <>
class Rooted<Value> {
  Heap& heap;
  size_t idx;

 public:
  Rooted(Heap& heap, Value obj) : heap(heap), idx(Heap::pushRoot(&heap, obj)) {}
  ~Rooted() { Heap::popRoot(&heap); }

  Value get() const { return Heap::getRoot(&heap, idx); }
  void set(Value obj) { Heap::setRoot(&heap, idx, obj); }
};

class Env : public HeapObject {
 public:
  Env* prev;
  Value val;

  static size_t offsetOfPrev() { return sizeof(HeapObject) + 0; }
  static size_t offsetOfVal() { return sizeof(HeapObject) + sizeof(Env*); }

  Env(Rooted<Env>& prev, Rooted<Value>& val)
      : HeapObject(Kind::Env), prev(prev.get()), val(val.get()) {}

  size_t byteSize() { return sizeof(*this); }
  void visitFields(Heap& heap) {
    heap.visit(&prev);
    val.visitFields(heap);
  }

  static Value lookup(Env* env, uint32_t depth) {
    fprintf(stderr, "(runtime) looking up %u\n", depth);
    while (depth--) {
      if (env == nullptr) {
        signal_error("Invalid depth -- too deep",
                     std::to_string(depth).c_str());
      }
      env = env->prev;
    }
    assert(env != nullptr);
    return env->val;
  }
};

class Closure : public HeapObject {
 public:
  Env* env;
  Func* func;

  static size_t offsetOfEnv() { return sizeof(HeapObject) + 0; }
  static size_t offsetOfFunc() { return sizeof(HeapObject) + sizeof(Env*); }

  Closure(Rooted<Env>& env, Func* func)
      : HeapObject(Kind::Closure), env(env.get()), func(func) {}

  size_t byteSize() { return sizeof(*this); }
  void visitFields(Heap& heap) { heap.visit(&env); }
};

HeapObject* Heap::copy(HeapObject* obj) {
  if (obj->isForwarded()) {
    signal_error("should not Heap::copy forwarded object", nullptr);
  }
  size_t size;
  switch (obj->kind()) {
#define COMPUTE_SIZE(name, Name)                                               \
  case HeapObject::Kind::Name:                                                 \
    size = obj->as##Name()->byteSize();                                        \
    break;
    FOR_EACH_HEAP_OBJECT_KIND(COMPUTE_SIZE)
#undef COMPUTE_SIZE
  }
  HeapObject* new_obj = reinterpret_cast<HeapObject*>(hp);
  memcpy(new_obj, obj, size);
  obj->forward(new_obj);
  hp += alignUp(size);
  return new_obj;
}

size_t Heap::scan(HeapObject* obj) {
  switch (obj->kind()) {
#define SCAN_OBJECT(name, Name)                                                \
  case HeapObject::Kind::Name:                                                 \
    obj->as##Name()->visitFields(*this);                                       \
    return obj->as##Name()->byteSize();
    FOR_EACH_HEAP_OBJECT_KIND(SCAN_OBJECT)
#undef SCAN_OBJECT
    default:
      abort();
  }
}

void Heap::visitRoots() {
  for (Value& root : roots) {
    root.visitFields(*this);
  }
}

static Value eval_primcall(Prim::Op op, intptr_t lhs, intptr_t rhs) {
  // FIXME: What to do on overflow.
  switch (op) {
    case Prim::Op::Eq:
      return Value(lhs == rhs);
    case Prim::Op::LessThan:
      return Value(lhs < rhs);
    case Prim::Op::Add:
      return Value(lhs + rhs);
    case Prim::Op::Sub:
      return Value(lhs - rhs);
    case Prim::Op::Mul:
      return Value(lhs * rhs);
    default:
      signal_error("unexpected primcall op", nullptr);
      return Value(nullptr);
  }
}

static std::set<Func*> jitCandidates;

typedef Value (*JitFunction)(Env*, Heap*);

static Value eval(Expr* expr, Env* unrooted_env, Heap& heap) {
  Rooted<Env> env(heap, unrooted_env);

tail:
  switch (expr->kind) {
    case Expr::Kind::Func: {
      Func* func = static_cast<Func*>(expr);
      if (!func->jitCode) {
        jitCandidates.insert(func);
      }
      return Value(new (heap) Closure(env, func));
    }
    case Expr::Kind::Var: {
      Var* var = static_cast<Var*>(expr);
      return Env::lookup(env.get(), var->depth);
    }
    case Expr::Kind::Prim: {
      Prim* prim = static_cast<Prim*>(expr);
      Value lhs = eval(prim->lhs.get(), env.get(), heap);
      if (!lhs.isSmi()) {
        signal_error("primcall expected integer lhs, got", lhs.kindName());
      }
      Value rhs = eval(prim->rhs.get(), env.get(), heap);
      if (!rhs.isSmi()) {
        signal_error("primcall expected integer rhs, got", rhs.kindName());
      }
      return eval_primcall(prim->op, lhs.getSmi(), rhs.getSmi());
    }
    case Expr::Kind::Literal: {
      Literal* literal = static_cast<Literal*>(expr);
      return Value(literal->val);
    }
    case Expr::Kind::Call: {
      Call* call = static_cast<Call*>(expr);
      Rooted<Value> func(heap, eval(call->func.get(), env.get(), heap));
      if (!func.get().isClosure()) {
        signal_error("call expected closure, got", func.get().kindName());
      }
      Closure* closure = func.get().asClosure();
      Rooted<Env> call_env(heap, closure->env);
      for (const std::unique_ptr<Expr>& arg : call->args) {
        // TODO(max): Hoist handle out of loop
        Rooted<Value> arg_val(heap, eval(arg.get(), env.get(), heap));
        Env* new_call_env = new (heap) Env(call_env, arg_val);
        call_env.set(new_call_env);
      }
      if (closure->func->jitCode) {
        JitFunction f = reinterpret_cast<JitFunction>(closure->func->jitCode);
        return f(call_env.get(), &heap);
      }
      expr = closure->func->body.get();
      env.set(call_env.get());
      goto tail;
    }
    case Expr::Kind::LetRec: {
      LetRec* letrec = static_cast<LetRec*>(expr);
      Rooted<Env> let_env(heap, env.get());
      for (const std::unique_ptr<Expr>& arg : letrec->args) {
        // TODO(max): Hoist handle out of loop
        Rooted<Value> arg_val(heap, eval(arg.get(), env.get(), heap));
        Rooted<Env> new_let_env(heap, new (heap) Env(let_env, arg_val));
        let_env.set(new_let_env.get());
      }
      return eval(letrec->body.get(), let_env.get(), heap);
      // env.set(let_env.get());
      // expr = letrec->body.get();
      // goto tail;
    }
    case Expr::Kind::If: {
      If* ifexpr = static_cast<If*>(expr);
      Value test = eval(ifexpr->test.get(), env.get(), heap);
      if (!test.isSmi()) {
        signal_error("if expected integer, got", test.kindName());
      }
      expr = test.getSmi() ? ifexpr->consequent.get() : ifexpr->alternate.get();
      goto tail;
    }
    default:
      signal_error("unexpected expr kind", nullptr);
      return Value(nullptr);
  }
}

EXPORT("eval")
Value eval(Expr* expr, size_t heap_size) {
  Heap heap(heap_size);
  return eval(expr, /*unrooted_env=*/nullptr, heap);
}

char assertEqual(const char* program, size_t heap_size, intptr_t expected) {
  std::unique_ptr<Expr> expr(std::move(parse(program)));
  Value result = eval(expr.get(), heap_size);
  if (!result.isSmi()) {
    fprintf(stderr, "Expected integer but got %s\n", result.kindName());
    return 'F';
  }
  if (result.getSmi() != expected) {
    fprintf(stderr, "Expected %ld but got %ld\n", expected, result.getSmi());
    return 'F';
  }
  return '.';
}

#ifdef LIBRARY
EXPORT("allocateBytes")
void* allocateBytes(size_t len) { return malloc(len); }
EXPORT("freeBytes")
void freeBytes(void* ptr) { free(ptr); }
#else
int main(int argc, char* argv[]) {
  struct {
    const char* program;
    intptr_t expected;
  } tests[] = {
      // Literals
      {"123", 123},
      // Primitives
      {"(+ 3 4)", 7},
      // Recursive eval
      {"(+ (+ 1 2) (- 7 3))", 7},
      // Let bindings and symbol lookup
      {"(letrec ((a 123)) a)", 123},
      // Let shadowing
      {"(letrec ((a 123)) (letrec ((a 456)) a))", 456},
      // Multiple let bindings and symbol lookup
      {"(letrec ((a 3) (b 4)) (+ a b))", 7},
      // Let binding order
      {"(letrec ((a 3) (b 4)) (+ a b))", 7},
      // No-parameter functions
      {"(letrec ((const (lambda () 3))) (const))", 3},
      // Single-parameter functions
      {"(letrec ((inc (lambda (x) (+ 1 x)))) (inc 3))", 4},
      // Multiple-parameter functions
      {"(letrec ((sub (lambda (a b) (- b a)))) (sub 4 3))", -1},
      {"(letrec ((sub (lambda (a b) (- a b)))) (sub 4 3))", 1},
      {"(letrec ((sub (lambda (b a) (- b a)))) (sub 4 3))", 1},
      {"(letrec ((sub (lambda (b a) (- a b)))) (sub 4 3))", -1},
      // Recursion
      {"(letrec ("
       "         (fac (lambda (x)"
       "                (if (< x 2)"
       "                  x"
       "                  (* x (fac (- x 1))))))"
       "         )"
       "  (fac 5))",
       120},
      // Multiple parameters
      {"(letrec ("
       "         (add (lambda (x y)"
       "                (+ x y)))"
       "         )"
       "  (add 3 4))",
       7},
      {nullptr, 0},
  };
  fprintf(stdout, "Running tests ");
  for (size_t i = 0; tests[i].program != nullptr; i++) {
  fprintf(stderr, "---\n");
    fputc(assertEqual(tests[i].program, /*heap_size=*/1024, tests[i].expected),
          stdout);
  }
  fprintf(stdout, "\n");
}
#endif
