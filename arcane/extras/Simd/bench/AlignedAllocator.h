#ifndef ARCANE_ALIGNED_ALLOCATOR
#define ARcANE_ALIGNED_ALLOCATOR

#include <cstddef>
#include <stdlib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Allocateur conforme à std::allocator du C++11 pour gérer l'alignement
// Cet allocateur peut être utilisé pour std::vector.
inline void*
ArcaneAlignedAlloc(std::size_t alignment,std::size_t size) ARCANE_NOEXCEPT
{
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }
  void* p;
  if (::posix_memalign(&p, alignment, size) != 0) {
    p = 0;
  }
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void ArcaneAlignedFree(void* ptr) ARCANE_NOEXCEPT
{
  ::free(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class AlignedAllocator
{
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef void* void_pointer;
  typedef const void* const_void_pointer;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T& reference;
  typedef const T& const_reference;

 public:
  template<class U>
  struct rebind {
    typedef AlignedAllocator<U> other;
  };
  
  AlignedAllocator() noexcept = default;

  template<class U>
  AlignedAllocator(const AlignedAllocator<U>&) ARCANE_NOEXCEPT
  { }

  pointer address(reference value) const ARCANE_NOEXCEPT
  {
    return std::addressof(value);
  }

  const_pointer address(const_reference value) const ARCANE_NOEXCEPT
  {
    return std::addressof(value);
  }

  pointer allocate(size_type size,const_void_pointer = 0)
  {
    void* p = ArcaneAlignedAlloc(64,sizeof(T) * size);
    if (!p && size > 0) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(p);
  }

  void deallocate(pointer ptr, size_type)
  {
    ArcaneAlignedFree(ptr);
  }
  // Taille max possible pour une allocation.
  constexpr size_type max_size() const ARCANE_NOEXCEPT
  {
    return ((size_type)2) << 40;
  }
  
  template<class U, class... Args> void construct(U* ptr, Args&&... args) {
    void* p = ptr;
    ::new(p) U(std::forward<Args>(args)...);
  }

  template<class U>
  void construct(U* ptr)
  {
    void* p = ptr;
    ::new(p) U();
  }

  template<class U>
  void destroy(U* ptr)
  {
    (void)ptr;
    ptr->~U();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T1, class T2> inline bool
operator==(const AlignedAllocator<T1>&,
           const AlignedAllocator<T2>&) ARCANE_NOEXCEPT
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T1, class T2> inline bool
operator!=(const AlignedAllocator<T1>&,
           const AlignedAllocator<T2>&) ARCANE_NOEXCEPT
{
  return false;
}



#endif
