/*---------------------------------------------------------------------------*/
/* Array.cc                                                    (C) 2000-2018 */
/*                                                                           */
/* Vecteur de données 1D.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/Iterator.h"

#include "arccore/collections/Array.h"

#include <algorithm>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase ArrayImplBase::shared_null_instance = ArrayImplBase();
ArrayImplBase* ArrayImplBase::shared_null = &ArrayImplBase::shared_null_instance;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BadAllocException
: public std::bad_alloc
{
 public:
  BadAllocException(const std::string& str) : m_message(str){}
  virtual const char* what() const ARCCORE_NOEXCEPT
  {
    return m_message.c_str();
  }
 public:
  std::string m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO: pour les allocations, faire en sorte que le
 * début du tableau soit aligné sur 16 octets dans tous les cas.
 * Attention dans ce cas a bien traiter les problèmes avec realloc().
 * TODO: pour les grosses allocations qui correspondantes la
 * plupart du temps à des variables, ajouter un random sur le début
 * du tableau pour éviter les conflits de bancs mémoire ou de cache
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* DefaultArrayAllocator::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,
         Int64 sizeof_true_type,ArrayImplBase* init)
{
  Int64 elem_size = sizeof_true_impl + (new_capacity - 1) * sizeof_true_type;
  /*std::cout << " ALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeofTypedData
            << " size=" << size << " datasize=" << sizeofT << '\n';*/
  ArrayImplBase* p = (ArrayImplBase*)::malloc(elem_size);
  //std::cout << " RETURN p=" << p << '\n';
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad DefaultArrayAllocator::allocate() size=" << elem_size << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type << '\n';
    throw BadAllocException(ostr.str());
  }
  Int64 s = (new_capacity>init->capacity) ? init->capacity : new_capacity;
  ::memcpy(p, init,sizeof_true_impl + (s - 1) * sizeof_true_type);
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultArrayAllocator::
deallocate(ArrayImplBase* ptr)
{
  ::free(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* DefaultArrayAllocator::
reallocate(Int64 sizeof_true_impl,Int64 new_capacity,
           Int64 sizeof_true_type,ArrayImplBase* ptr)
{
  Int64 elem_size = sizeof_true_impl + (new_capacity - 1) * sizeof_true_type;
  //Integer elem_size = sizeofTypedData + (size - 1) * sizeofT;
  std::cout << " REALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " ptr=" << ptr << '\n';
  ArrayImplBase* p = (ArrayImplBase*)::realloc(ptr,elem_size);
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad DefaultArrayAllocator::reallocate() size=" << elem_size << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type
         << " old_ptr=" << ptr << '\n';
    throw BadAllocException(ostr.str());
  }
  //std::cout << " RETURN p=" << ((Int64)p%16) << '\n';
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 DefaultArrayAllocator::
computeCapacity(Int64 current,Int64 wanted)
{
  Int64 capacity = current;
  //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
  while (wanted>capacity)
    capacity = (capacity==0) ? 4 : (capacity + 1 + capacity / 2);
  //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
  return capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,
         Int64 sizeof_true_type,ArrayImplBase* init)
{
  return allocate(sizeof_true_impl,new_capacity,sizeof_true_type,init,nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,
         Int64 sizeof_true_type,ArrayImplBase* init,IMemoryAllocator* allocator)
{
  if (!allocator)
    allocator = init->allocator;

  size_t s_sizeof_true_impl = (size_t)sizeof_true_impl;
  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = allocator->adjustCapacity(s_new_capacity,sizeof_true_type);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_sizeof_true_impl + (s_new_capacity - 1) * s_sizeof_true_type;
  ArrayImplBase* p = (ArrayImplBase*)(allocator->allocate(elem_size));
#ifdef ARCCORE_DEBUG_ARRAY
  std::cout << "ArrayImplBase::ALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " p=" << p << '\n';
#endif
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad ArrayImplBase::allocate() size=" << elem_size << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type << '\n';
    throw BadAllocException(ostr.str());
  }

  *p = *init;

  p->capacity = (Int64)s_new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
reallocate(Int64 sizeof_true_impl,Int64 new_capacity,Int64 sizeof_true_type,
           ArrayImplBase* current)
{
  IMemoryAllocator* allocator = current->allocator;
  size_t s_sizeof_true_impl = (size_t)sizeof_true_impl;
  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = allocator->adjustCapacity(s_new_capacity,sizeof_true_type);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_sizeof_true_impl + (s_new_capacity - 1) * s_sizeof_true_type;
  
  ArrayImplBase* p = 0;
  {
    const bool use_realloc = allocator->hasRealloc();
    // Lorsqu'on voudra implémenter un realloc avec alignement, il faut passer
    // par use_realloc = false car sous Linux il n'existe pas de méthode realloc
    // garantissant l'alignmenent (alors que sous Win32 si :) ).
    // use_realloc = false;
    if (use_realloc){
      p = (ArrayImplBase*)(allocator->reallocate(current,elem_size));
    }
    else{
      p = (ArrayImplBase*)(allocator->allocate(elem_size));
      //GG: TODO: regarder si 'current' peut etre nul (a priori je ne pense pas...)
      if (p && current){
        size_t current_size = s_sizeof_true_impl + (current->size - 1) * s_sizeof_true_type;
        ::memcpy(p,current,current_size);
        allocator->deallocate(current);
      }
    }
  }
#ifdef ARCCORE_DEBUG_ARRAY
  std::cout << " ArrayImplBase::REALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " ptr=" << current << " new_p=" << p << '\n';
#endif
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad ArrayImplBase::reallocate() size=" << elem_size
         << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type
         << " old_ptr=" << current << '\n';
    throw BadAllocException(ostr.str());
  }
  p->capacity = (Int64)s_new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayImplBase::
deallocate(ArrayImplBase* current)
{
  current->allocator->deallocate(current);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayImplBase::
overlapError(const void* begin1,Int64 size1,
             const void* begin2,Int64 size2)
{
  ARCCORE_UNUSED(begin1);
  ARCCORE_UNUSED(begin2);
  ARCCORE_UNUSED(size1);
  ARCCORE_UNUSED(size2);
  throw FatalErrorException(A_FUNCINFO,"source and destinations overlaps");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayImplBase::
throwBadSharedNull()
{
  throw BadAllocException("corrupted ArrayImplBase::shared_null");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_UT_CHECK(expr,message) \
if ( ! (expr) )\
  throw FatalErrorException((message))

class IntSubClass
{
public:
  IntSubClass(Integer v) : m_v(v) {}
  IntSubClass() : m_v(0) {}
  Integer m_v;
  bool operator==(Integer iv) const { return m_v==iv; }
};
ARCCORE_DEFINE_ARRAY_PODTYPE(IntSubClass);

class IntPtrSubClass
{
public:
  static Integer count;
  IntPtrSubClass(Integer v) : m_v(new Integer(v)) { ++count; }
  IntPtrSubClass() : m_v(new Integer(0)) { ++count; }
  ~IntPtrSubClass() { --count; delete m_v; }
  Integer* m_v;
  IntPtrSubClass(const IntPtrSubClass& v) : m_v(new Integer(*v.m_v)) { ++count; }
  void operator=(const IntPtrSubClass& v)
    {
      Integer* n = new Integer(*v.m_v);
      delete m_v;
      m_v = n;
    }
  bool operator==(Integer iv) const
  {
    //cout << "** COMPARE: " << *m_v << " v=" << iv << '\n';
    return *m_v==iv;
  }
};
Integer IntPtrSubClass::count = 0;

template<typename Container,typename SubClass>
class IntegerArrayTester
{
public:
  void test()
  {
    {
      Container c;
      c.add(SubClass(1));
      c.add(SubClass(2));
      c.add(SubClass(3));
      ARCCORE_UT_CHECK((c.size()==3),"Bad size (3)");
      ARCCORE_UT_CHECK((c[0]==1),"Bad value [0]");
      ARCCORE_UT_CHECK((c[1]==2),"Bad value [1]");
      ARCCORE_UT_CHECK((c[2]==3),"Bad value [2]");
      c.resize(0);
      ARCCORE_UT_CHECK((c.size()==0),"Bad size (0)");
      c.resize(5);
      ARCCORE_UT_CHECK((c.size()==5),"Bad size");
      c.add(SubClass(6));
      ARCCORE_UT_CHECK((c.size()==6),"Bad size");
      ARCCORE_UT_CHECK((c[5]==6),"Bad value [5]");
    }
    {
      Container c;
      for( Integer i=0; i<50; ++i )
        c.add(SubClass(i+2));
      {
        Container c2 = c;
        for( Integer i=50; i<100; ++i ){
          c2.add(SubClass(i+2));
        }
        Container c4 = c2;
        Container c3;
        c3.add(5);
        c3.add(6);
        c3.add(7);
        c3 = c2;
        for( Integer i=100; i<150; ++i ){
          c4.add(SubClass(i+2));
        }
      }
      ARCCORE_UT_CHECK((c.size()==150),"Bad size (150)");
      c.reserve(300);
      ARCCORE_UT_CHECK((c.capacity()==300),"Bad capacity (300)");
      for( Integer i=0; i<50; ++i ){
        c.remove(i);
      }
      ARCCORE_UT_CHECK((c.size()==100),"Bad size (100)");
      for( Integer i=0; i<50; ++i ){
        //cout << "** VAL: " << i << " c=" << c[i] << " expected=" << ((i*2)+3) << '\n';
        ARCCORE_UT_CHECK((c[i]==((i*2)+3)),"Bad value");
      }
      for( Integer i=50; i<100; ++i ){
        //cout << "** VAL: " << i << " c=" << c[i] << " expected=" << (i+52) << '\n';
        ARCCORE_UT_CHECK((c[i]==(i+52)),"Bad value");
      }
    }
  }
};

void
_testArrayNewInternal()
{
  std::cout << "** TEST VECTOR NEW\n";

  size_t impl_size = sizeof(ArrayImplBase);
  size_t wanted_size = AlignedMemoryAllocator::simdAlignment();
  std::cout << "** sizeof(ArrayImplBase) = " << impl_size << '\n';
  if (impl_size!=wanted_size)
    ARCCORE_FATAL("Bad sizeof(ArrayImplBase) v={0} expected={1}",impl_size,wanted_size);
  {
    IntegerArrayTester< SharedArray<IntSubClass>, IntSubClass > rvt;
    rvt.test();
  }
  std::cout << "** TEST VECTOR NEW 2\n";
  {
    IntegerArrayTester< SharedArray<IntPtrSubClass>, IntPtrSubClass > rvt;
    rvt.test();
    std::cout << "** COUNT = " << IntPtrSubClass::count << "\n";

  }
  std::cout << "** TEST VECTOR NEW 3\n";
  {
    IntegerArrayTester< SharedArray<IntPtrSubClass>, IntPtrSubClass > rvt;
    rvt.test();
    std::cout << "** COUNT = " << IntPtrSubClass::count << "\n";

  }
  {
    SharedArray<IntSubClass> c;
    c.add(5);
    c.add(7);
    SharedArray<IntSubClass> c2(c.clone());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c;
    c.add(5);
    c.add(7);
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c { 5, 7 };
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c { 5, 7 };
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
  }
  {
    SharedArray<IntSubClass> c { 5, 7 };
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
  }
  {
    PrintableMemoryAllocator allocator;
    UniqueArray<IntSubClass> c(&allocator);
    UniqueArray<IntSubClass> cx(&allocator,5);
    ARCCORE_UT_CHECK((cx.size()==5),"Bad value [5]");
    c.add(5);
    c.add(7);
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
    for( Integer i=0; i<50; ++i )
      c.add(i+3);
    c.resize(24);
    c.reserve(70);
  }
  {
    SharedArray<IntSubClass> c2;
    c2.add(3);
    c2.add(5);
    {
      SharedArray<IntSubClass> c;
      c.add(5);
      c.add(7);
      c2 = c;
    }
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<Int32> values1 = { 2, 5 };
    UniqueArray<Int32> values2 = { 4, 9, 7 };
    // Copie les valeurs de values2 à la fin de values1.
    std::copy(std::begin(values2),std::end(values2),std::back_inserter(values1));
    std::cout << "** VALUES1 = " << values1 << "\n";
    ARCCORE_UT_CHECK((values1.size()==5),"BI: Bad size");
    ARCCORE_UT_CHECK((values1[0]==2),"BI: Bad value [0]");
    ARCCORE_UT_CHECK((values1[1]==5),"BI: Bad value [1]");
    ARCCORE_UT_CHECK((values1[2]==4),"BI: Bad value [2]");
    ARCCORE_UT_CHECK((values1[3]==9),"BI: Bad value [3]");
    ARCCORE_UT_CHECK((values1[4]==7),"BI: Bad value [4]");

    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    UniqueArray<IntPtrSubClass>::iterator i = std::begin(vx);
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << '\n';
  }
  {
    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    auto range = vx.range();
    UniqueArray<IntPtrSubClass>::iterator i = vx.begin();
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << '\n';
    const UniqueArray<IntPtrSubClass>& cvx = (const UniqueArray<IntPtrSubClass>&)(vx);
    UniqueArray<IntPtrSubClass>::const_iterator cicvx = cvx.begin();
    std::cout << "V=" << cicvx->m_v << '\n';
  }
  {
    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    auto r = vx.range();
    UniqueArray<IntPtrSubClass>::iterator i = std::begin(vx);
    UniqueArray<IntPtrSubClass>::iterator iend = std::end(vx);
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << " " << (iend-i) << '\n';
    const UniqueArray<IntPtrSubClass>& cvx = (const UniqueArray<IntPtrSubClass>&)(vx);
    UniqueArray<IntPtrSubClass>::const_iterator cicvx = std::begin(cvx);
    std::cout << "V=" << cicvx->m_v << '\n';
    std::copy(std::begin(vx),std::end(vx),std::begin(vx));
  }
  {
    UniqueArray<Int32> values = { 4, 9, 7 };
    for( typename ArrayView<Int32>::const_iter i(values.view()); i(); ++i ){
      std::cout << *i << '\n';
    }
    for( typename ConstArrayView<Int32>::const_iter i(values.view()); i(); ++i ){
      std::cout << *i << '\n';
    }
    for( auto i : values.range()){
      std::cout << i << '\n';
    }
    for( auto i : values.constView().range()) {
      std::cout << i << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
