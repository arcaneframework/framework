# Array Views in Arccore {#arcanedoc_core_types_array_views}

[TOC]

This document describes the four non-owning array view types provided by the
`arccore` **base** component and explains when to use each of them.

- \arcane{ArrayView} / \arcane{ConstArrayView} — `arccore/base/ArrayView.h`.
Those are the **original %Arcane API** (2000s). They are used pervasively in
the public Arcane framework API, e.g. mesh variables
(`ArrayView<double>` returned by `CellVariable<double>::values()`), mesh readers, I/O, etc.
- \arcane{Span} / \arcane{SmallSpan} — `arccore/base/Span.h`

You can also add forward declarations for these classes using the header file
arccore/base/BaseTypes.h`.

---

## 1. Common properties

All four classes implement the same concept: a **lightweight, non-owning view
over a contiguous block of memory**, similar to a C array with a size.

- **No memory management.** The view only stores a pointer and a size. The
  memory is owned by a container (\arcane{Array}, \arcane{UniqueArray}, \arcane{SharedArray}, \arcane{NumArray}) or by
  a raw buffer. A view is only valid as long as the underlying container is
  not reallocated.
- **Cheap copies.** Constructors and assignment operators copy only the
  pointer and size — never the data.
- **Contiguity guaranteed.** All elements are consecutive in memory, so a view
  can be passed to C APIs or reinterpreted as bytes (\arcane{asBytes()}).
- **Optional bounds checking.** When arccore is compiled in check mode
  (`ARCCORE_CHECK` defined, i.e. `ARCCORE_BUILD_MODE=Debug`/`Check`),
  out-of-bounds access through `operator[]`/`at()` throws
  `IndexOutOfRangeException`.
- **`std::array` interop.** All four types can be constructed from (and
  assigned from) `std::array<T, N>`.
- **Common accessors** (exact signature depends on the type):

  | Member | Description |
  |---|---|
  | `size()` / `length()` | Number of elements |
  | `empty()` | `size() == 0` |
  | `data()` / `unguardedBasePointer()` | Pointer to the first element |
  | `operator[]`, `operator()`, `item(i)`, `at(i)` | Element access (checked in check mode) |
  | `setItem(i, v)` / `setAt(i, v)` | Element write (writable views only) |
  | `begin()` / `end()` | Iterators — usable with range-based `for` |
  | `subView()` / `subSpan()` / `subPart()` / `subspan()` | Sub-view `[abegin, abegin+asize)`, truncated to the view size |
  | `contains(v)` | Linear search |
  | `copy(other)` | Element-wise copy from a compatible view |
  | `fill(v)` | Set all elements to `v` |

\note **Note on naming.** The Span family provides both `subspan()` (C++20
`std::span` spelling) and `subSpan()`/`subPart()` (%Arcane spelling) with
identical behavior. The old `subView()` spelling is **deprecated**
(`ARCCORE_DEPRECATED_REASON("Y2023: use subSpan() instead")`).

---

## 2. `ArrayView<T>` — the classic writable view

The class \arcane{ArrayView} is defined in `arccore/base/ArrayView.h`.

```cpp
template <class T>
class ArrayView
{
  // ...
  Integer m_size; //!< Number of elements
  T*      m_ptr;  //!< Pointer to the array
};
```

- **Writable**: \arcane{ArrayView::operator[]()} returns `T&`, \arcane{ArrayView::data()} returns `T*`.
- **Size type is `Integer`**, which is `Int32`. The maximum number of elements is therefore
  ~2.1 × 10⁹ (`Int32`).
- **Layout**: two data members (`Integer m_size; T* m_ptr;`) → 16 bytes in the
  default build.
- **Host code only** — no `ARCCORE_HOST_DEVICE` annotations, so it is not part
  of the accelerator (GPU) API surface.

### Typical API

```cpp
ArrayView<T>            subView(Integer abegin, Integer asize);
ConstArrayView<T>       constView() const;
ConstArrayView<T>       subConstView(Integer abegin, Integer asize) const;
ArrayView<T>            subViewInterval(Integer index, Integer nb_interval);
void                    setArray(const ArrayView<T>& v);  // replace ptr+size
void                    copy(const U& other);
```

### Example

```cpp
Real t[5];
ArrayView<Real> a(t, 5);   // view over a C array (no copy)
a[2] = 5.0;                // writable
Real sum = 0.0;
for (Real v : a.subView(0, 3))  // first 3 elements
  sum += v;
```

---

## 3. `ConstArrayView<T>` — the classic read-only view

The class \arcane{ConstArrayView} is defined in `arccore/base/ArrayView.h`.

```cpp
template <class T>
class ConstArrayView
{
  // ...
  Integer           m_size; //!< Number of elements
  const T*          m_ptr;  //!< Pointer to the array (read-only)
};
```

- Identical to \arcane{ArrayView<T>} **except** that it exposes only `const T*`
  pointers, const references and const iterators. Elements cannot be modified
  through the view.
- **Implicit conversion from `ArrayView<T>`** (constructor and assignment) — a safe "upcast" that copies only ptr+size:

  ```cpp
  ArrayView<Real>      v;
  ConstArrayView<Real> cv = v;   // OK, implicit
  ```

- Same size semantics as \arcane{ArrayView} (`Integer`).
- This is the return type of const accessors throughout the %Arcane API
  (e.g. `const` mesh variable accessors).

---

## 4. `Span<T, Extent>` — the C++20-style, Int64-sized view

The class \arcane{Span} is defined in `arccore/base/Span.h`.

```cpp
template <typename T, Int64 Extent = DynExtent>
class Span : public SpanImpl<T, Int64, Extent> { ... };
```

It is designed to be similar to the C++20 `std::span` class,
with %Arcane specific additions:

- **Size stored as `Int64`** — can address arrays with more than 2³¹
  elements, unlike \arcane{ArrayView}.
- **Compile-time extent.** `Extent` defaults to `DynExtent` (`-1`, size known
  at runtime). If a positive `Extent` is given, the size is a compile-time
  constant:

  - the size member is not stored (empty `ExtentStorage`),
    so a fixed-extent span is only **8 bytes** (the pointer);
  - the constructor `Span(T* ptr)` (no size argument) becomes available
    (`requires(!IsDynamic)`);
  - in check mode, constructing with a mismatched runtime size throws.
- **`ARCCORE_HOST_DEVICE`** on all accessors — the type is part of the
  **accelerator API** and can be used in CUDA/HIP/SYCL device code.
- **Const-ness via the element type** (like `std::span`): `Span<T>` is
  writable, `Span<const T>` is read-only. There is no separate
  "ConstSpan" class.

### Conversions (all implicit, ptr+size copies)

| From | To | Condition |
|---|---|---|
| `ArrayView<X>` | `Span<X>` / `SmallSpan<X>` | — |
| `ConstArrayView<X>` | `Span<const X>` / `SmallSpan<const X>` | `T` must be `const X` |
| `Span<X>` / `SmallSpan<X>` | `Span<const X>` / `SmallSpan<const X>` | `T` must be `const X` |
| `std::array<X, N>` | any span of `X` or `const X` | — |

`view_type` maps each span back to the legacy type: `Span<T>::view_type` is
`ArrayView<T>`, `Span<const T>::view_type` is `ConstArrayView<T>`.
The helpers `smallView()` / `constSmallView()` perform that
conversion explicitly.

### Typical API

```cpp
Int64   size() const;
Int64   sizeBytes() const;                    // size * sizeof(T)
Span<T, DynExtent> subspan(Int64 abegin, Int64 asize) const;   // std::span spelling
Span<T, DynExtent> subSpan(Int64 abegin, Int64 asize) const;   // Arcane spelling
Span<T, DynExtent> subPart(Int64 abegin, Int64 asize) const;
Span<T, DynExtent> subSpanInterval(Int64 index, Int64 nb_interval) const;
ArrayView<T> smallView();                     // downconvert to legacy type
```

### Example

```cpp
// Dynamic extent (default)
Span<Real> s = ArrayView<Real>(ptr, 1000);
s.subspan(10, 20)[0] = 1.0;

// Fixed extent: size known at compile time, no size stored
Span<Real, 3> v3{ &buffer[0] };   // requires 8-byte alignment of Extent == 3
static_assert(sizeof(v3) == 8);

// Read-only view
Span<const Real> cs = s;
```

---

## 5. `SmallSpan<T, Extent>` — the Int32-sized counterpart

The class \arcane{SmallSpan} is defined in `arccore/base/Span.h` (inherits from
`SpanImpl<T, Int32, Extent>`).

```cpp
template <typename T, Int32 Extent = DynExtent>
class SmallSpan : public SpanImpl<T, Int32, Extent> { ... };
```

- **Identical API and semantics as `Span`**, except the size is stored as
  **`Int32`** (forward declarations: `BaseTypes.h:65`).
- The class documentation adds a constraint: *the number of bytes associated
  with the view (`sizeBytes()`) must also fit within an `Int32`* — i.e. a
  `SmallSpan` of 1-byte elements cannot exceed ~2 GB of data.
- Same features as `Span`: fixed or dynamic extent, `ARCCORE_HOST_DEVICE`,
  `std::span`-like conversions, deprecated `subView()` spelling, etc.
- **Helper to pick the span type from a size type**:

  ```cpp
  template <typename T, typename SizeType>
  class SpanTypeFromSize;
  // SpanTypeFromSize<T, Int32>::SpanType == SmallSpan<T>
  // SpanTypeFromSize<T, Int64>::SpanType == Span<T>
  ```

  This is used, for example, by `asBytes()` to return the span type matching
  the source size type.

### When to prefer `SmallSpan` over `Span`

- The data is small enough for `Int32` sizing (the common case: mesh entity
  lists, cell/node variables, buffers).
- Interoperating with the legacy API: `ArrayView`'s size is `Integer`
  (= `Int32` in the default build), so `ArrayView ↔ SmallSpan` round-trips
  without any width change.
- It is the natural choice for fixed-extent small buffers
  (`SmallSpan<Real, 3>` for a vector, `SmallSpan<Real, 9>` for a 3×3 matrix,
  ...).

---

## 6. Comparison table

| | \arcane{ArrayView<T>} | \arcane{ConstArrayView<T>} | \arcane{SmallSpan<T, Extent>} | \arcane{Span<T, Extent>} |
|---|---|---|---|---|
| Header | `arccore/base/ArrayView.h` | `arccore/base/ArrayView.h` | `arccore/base/Span.h` | `arccore/base/Span.h` |
| Generation | Legacy (2000s) | Legacy (2000s) | New (C++20 era) | New (C++20 era) |
| Writable | Yes | No | `T` non-const | `T` non-const |
| Size type | `Integer` (`Int32` default, `Int64` with `ARCCORE_64BIT`) | `Integer` | `Int32` | `Int64` |
| Max elements (default build) | ~2.1 × 10⁹ | ~2.1 × 10⁹ | ~2.1 × 10⁹ (and `sizeBytes()` ≤ `Int32`) | ~9.2 × 10¹⁸ |
| Compile-time extent | No | No | Yes (`Extent` parameter) | Yes (`Extent` parameter) |
| Size member stored | Always | Always | Only if dynamic | Only if dynamic |
| `ARCCORE_HOST_DEVICE` (GPU) | No | No | Yes | Yes |
| `std::span`-like API | No | No | Yes | Yes |
| Const mechanism | Separate class | Separate class | `const` in element type | `const` in element type |
| Typical use | Legacy %Arcane public API (mesh variables, I/O) | Return of const accessors | New host/GPU code, Int32-sized data | New host/GPU code, potentially huge data |

---

## 7. Byte-level helpers (from `Span.h`)

These free functions bridge views and raw bytes (useful for serialization and
`memcpy`-style operations):

```cpp
SmallSpan<const std::byte> asBytes(const ArrayView<T>& s);          // read-only bytes
Span<const std::byte>      asBytes(const SpanImpl<T, SizeType, E>& s);
SmallSpan<std::byte>       asWritableBytes(const ArrayView<T>& s);  // writable
Span<std::byte>            asWritableBytes(const SpanImpl<...>& s); // T non-const
Span<T>                    asSpan(Span<std::byte, E> bytes);        // bytes -> typed
SmallSpan<T>               asSmallSpan(SmallSpan<std::byte, E> bytes);
Span<T, N>                 asSpan(std::array<T, N>& s);             // std::array -> span
SmallSpan<T, N>            asSmallSpan(std::array<T, N>& s);

void binaryWrite(std::ostream& ostr, const Span<const std::byte>& bytes);
void binaryRead (std::istream& istr, const Span<std::byte>& bytes);
```

\arcane{asBytes()} returns the span type (Small/Span) matching the source's size
type via `SpanTypeFromSize`.

---

## 8. Which type should I use?

1. **Calling existing %Arcane public APIs** (mesh variables, \arcane{ItemGroup}, I/O
   services, ...): use what the API takes or returns — usually
   \arcane{ArrayView} / \arcane{ConstArrayView}.
2. **Writing new code**, especially accelerator/GPU code: prefer the Span
   family (`ARCCORE_HOST_DEVICE`).
   - Default: **`SmallSpan<T>`** (covers all common mesh-data sizes and
     round-trips losslessly with `ArrayView`).
   - Use **`Span<T>`** when the element count or byte size may exceed `Int32`
     limits.
   - Use a **fixed extent** (`SmallSpan<T, N>` / `Span<T, N>`) for
     compile-time-known sizes (vectors, matrices, small fixed buffers) —
     the object then contains only the pointer.
3. **Never** treat a view as an owner: do not delete/free its `data()` and
   keep the view alive no longer than the container it references.
4. **Migrating legacy code**: replace `subView()` by `subSpan()`/`subPart()`;
   `subView()` still compiles but is deprecated.

### Quick example: legacy vs. new style

```cpp
// Legacy style (still required by most Arcane public APIs)
void f(const ConstArrayView<Real>& values)
{
  for (Real v : values) { /* ... */ }
}

// New style (works on host and device)
void g(SmallSpan<const Real> values)
{
  Real s = 0.0;
  for (Real v : values) s += v;
}

// Interop: implicit, zero-cost conversion
ArrayView<Real> av;            // e.g. from a mesh variable
g(av);                         // ArrayView -> SmallSpan<const Real>
f(SmallSpan<const Real>(av));  // SmallSpan -> ConstArrayView
```
____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_numarray
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span> -->
</div>
