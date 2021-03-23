// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayUnitTest.cc                                            (C) 2000-2018 */
/*                                                                           */
/* Service de test des tableaux.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/Array3View.h"
#include "arcane/utils/Array4View.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"
#include "arcane/Timer.h"
#include "arcane/ServiceFinder2.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class Wrapper
{
 public:
  static Integer nb_new;
 public:
  Wrapper()
  : m_value(new DataType)
    {
      (*m_value) = DataType();
      ++nb_new;
    }
  Wrapper(DataType data)
  : m_value(new DataType(data))
    {
      ++nb_new;
    }
  Wrapper(const Wrapper<DataType>& rhs)
  : m_value(new DataType(rhs.value()))
    {
      ++nb_new;
    }
  ~Wrapper()
    {
      delete m_value;
      --nb_new;
    }
 public:
  operator DataType ()
    {
      return *m_value;
    }
  void operator=(DataType v)
    {
      *m_value = v;
    }
  void operator=(const Wrapper<DataType>& v)
    {
      *m_value = v.value();
    }
  bool operator==(const Wrapper<DataType>& rhs) const
    {
      return (*m_value) == (*rhs.m_value);
    }
  bool operator==(const DataType& rhs) const
    {
      return (*m_value) == rhs;
    }
  DataType value() const { return *m_value; }
 private:
  DataType* m_value;
};

template<typename DataType> Integer Wrapper<DataType>::nb_new = 0;

template<class T> inline ostream&
operator<<(ostream& o,const Wrapper<T>& v)
{
  o << v.value();
  return o;
}

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class Array2UnitTest
: public TraceAccessor
{
 public:
  Array2UnitTest(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

  void _printArray(const Array2<DataType>& array)
    {
      OStringStream ostr;
      Integer n1 = array.dim1Size();
      Integer n2 = array.dim2Size();
      ostr() << " Dim1=" << n1
             << " Dim2=" << n2
             << "\n";
      for( Integer i=0; i<n1; ++i ){
        ostr() << " I=[" << i << "] ->";
        for( Integer j=0; j<n2; ++j )
          ostr() << " [" << j << "]=" << array[i][j];
        ostr() << "\n";
      }
      ostr() << "\n";
      info() << ostr.str();
    }
  void _setArray(Array2<DataType>& array)
    {
      Integer n1 = array.dim1Size();
      Integer n2 = array.dim2Size();
      for( Integer i=0; i<n1; ++i )
        for( Integer j=0; j<n2; ++j )
          array[i][j] = (double)((i*n2)+j);
    }
  void _checkArray(Array2<DataType>& array,Integer original_dim2,Integer dim1,Integer dim2)
    {
      double total = DataType();
      for( Integer i=0; i<dim1; ++i )
        for( Integer j=0; j<dim2; ++j ){
          double v = (i*original_dim2)+j;
          info() << "VALUE i=" << i << " j=" << j << " v=" << v << " a=" << array[i][j];
          _check(array[i][j]==DataType(v),"Bad value");
          total += array[i][j];
        }
      info() << "TOTAL = " << total;
    }
  void doTest()
  {
    {
      info() << "Test 1";
      UniqueArray2<DataType> array;
      info() << array.dim2Size();
      array.resize(5);
    }
    {
      info() << "Test 2";
      UniqueArray2<DataType> array;
      info() << array.dim1Size();
      info() << array.dim2Size();
      array.resize(5,2);
      _check(array.dim1Size()==5,"Bad dim1size");
      _check(array.dim2Size()==2,"Bad dim2size");
      _printArray(array);
    }
    {
      info() << "Test 3";
      UniqueArray2<DataType> array(5,2);
      info() << "D1=" << array.dim1Size() << " D2=" << array.dim2Size();
      _printArray(array);
      _check(array.dim1Size()==5,"Bad dim1size");
      _check(array.dim2Size()==2,"Bad dim2size");
    }
    {
      info() << "Test 4";
      UniqueArray2<DataType> array;
      array.resize(5,3);
      info() << "D1=" << array.dim1Size() << " D2=" << array.dim2Size();
      _check(array.dim1Size()==5,"Bad size");
      _check(array.dim2Size()==3,"Bad dim2size");
      _setArray(array);
      _checkArray(array,3,5,3);
      
      // Teste reduction de taille
      array.resize(7,2);
      info() << "D1=" << array.dim1Size() << " D2=" << array.dim2Size();
      _check(array.dim1Size()==7,"Bad size (1)");
      _check(array.dim2Size()==2,"Bad dim2size (1)");
      _checkArray(array,3,5,2);
      info() << " NB NEW1=" << Wrapper<Real>::nb_new;

      // Teste augmentation de taille
      array.resize(9,5);
      info() << "D1=" << array.dim1Size() << " D2=" << array.dim2Size();
      _printArray(array);
      _check(array.dim1Size()==9,"Bad size (2)");
      _check(array.dim2Size()==5,"Bad dim2size (2)");
      _checkArray(array,3,5,2);
      info() << " NB NEW2=" << Wrapper<Real>::nb_new;

      // Teste reduction de taille de la premiere dimension
      array.resize(7,5);
      info() << "D1=" << array.dim1Size() << " D2=" << array.dim2Size();
      _check(array.dim1Size()==7,"Bad size (3)");
      _check(array.dim2Size()==5,"Bad dim2size (3)");
      _checkArray(array,3,5,2);
      info() << " NB NEW3=" << Wrapper<Real>::nb_new;
    }

    {
      info() << "Test 5.1";
      UniqueArray2<DataType> array;
      info() << array.dim2Size();
      array.resize(0,5);
      _check(array.dim1Size()==0,"Bad size (3)");
      _check(array.dim2Size()==5,"Bad dim2size (3)");
      array.resize(3,5);
      _printArray(array);
      array.add(DataType(6.3));
      _printArray(array);
      for( Integer j=0; j<5; ++j ){
        info() << "j=" << j << " v=" << array[3][j];
        _check(array[3][j]==ARCANE_REAL(6.3),"Bad value (4)");
      }
    }

    {
      // Vérifie que la copie se passe bien si uniquement la
      // première dimension est allouée.
      info() << "Test 5.2";
      UniqueArray2<DataType> array;
      info() << array.dim2Size();
      array.resize(50,0);
      _check(array.dim1Size()==50,"Bad size (4.1)");
      _check(array.dim2Size()==0,"Bad dim2size (4.1)");
      UniqueArray2<DataType> array2(array);
      _check(array2.dim1Size()==50,"Bad size (4.2)");
      _check(array2.dim2Size()==0,"Bad dim2size (4.2)");
      UniqueArray2<DataType> array3;
      array3 = array;
      _check(array3.dim1Size()==50,"Bad size (4.3)");
      _check(array3.dim2Size()==0,"Bad dim2size (4.3)");
      UniqueArray2<DataType> array4;
      array4.copy(array);
      _check(array4.dim1Size()==50,"Bad size (4.4)");
      _check(array4.dim2Size()==0,"Bad dim2size (4.4)");
    }

    // Vérifie qu'on n'a pas touché à ArrayImplBase::shared_null
    {
      info() << "Test 5.3";
      UniqueArray2<DataType> array;
      info() << array.dim1Size() << " " << array.dim2Size();
      _check(array.dim1Size()==0,"Bad size (5)");
      _check(array.dim2Size()==0,"Bad dim2size (5)");
    }
#ifndef ARCCORE_AVOID_DEPRECATED_ARRAY_CONSTRUCTOR
    {
      info() << "Test 6";
      Array2<DataType> array;
      info() << array.dim2Size();
      array.resizeNoInit(0,5);
      _check(array.dim1Size()==0,"Bad size (3)");
      _check(array.dim2Size()==5,"Bad dim2size (3)");
      array.resizeNoInit(3,5);
      _printArray(array);
      array.add(DataType(6.3));
      _printArray(array);
      for( Integer j=0; j<5; ++j ){
        info() << "j=" << j << " v=" << array[3][j];
        _check(array[3][j]==ARCANE_REAL(6.3),"Bad value (4)");
      }
    }
#endif
    // Test clone
#ifdef ARCANE_ALLOW_DEPRECATED_ARRAY2_COPY
    {
      UniqueArray2<DataType> array;
      array.resize(3,5);
      _fill(array);
      {
        UniqueArray2<DataType> cloned_array(array.clone());
        _checkSame(array,cloned_array);
      }
      array = UniqueArray2<DataType>();
      info() << "ARRAY SIZE=" << array.dim1Size() << " " << array.dim2Size();
      UniqueArray2<DataType> new_array;
      info() << "NEW ARRAY SIZE=" << new_array.dim1Size() << " " << new_array.dim2Size();
      _check(array.dim1Size()==0,"Bad size (test clone)");
    }
    // Test copy
    {
      UniqueArray2<DataType> array;
      array.resize(3,5);
      _fill(array);
      {
        UniqueArray2<DataType> copy_array;
        copy_array.copy(array);
        _checkSame(array,copy_array);
      }
    }
#endif
  }
  void _fill(Array2<DataType>& array)
  {
    Integer dim1_size = array.dim1Size();
    Integer dim2_size = array.dim2Size();
    for( Integer i=0; i<dim1_size; ++i ){
      for( Integer j=0; j<dim2_size; ++j ){
        array[i][j] = (DataType)(i*j);
      }
    }
  }
  void _checkSame(Array2<DataType>& array1,Array2<DataType>& array2)
  {
    Integer dim1_size = array1.dim1Size();
    Integer dim2_size = array1.dim2Size();
    Integer dim1_size_2 = array2.dim1Size();
    Integer dim2_size_2 = array2.dim2Size();
    _check(dim1_size==dim1_size_2,"Bad same size");
    _check(dim2_size==dim2_size_2,"Bad same size");
    for( Integer i=0; i<dim1_size; ++i ){
      for( Integer j=0; j<dim2_size; ++j ){
        _check(array1[i][j]==array2[i][j],"Bad value");
      }
    }
  }
  void _check(bool expression,const String& message)
    {
      if (!expression)
        throw FatalErrorException("Array2UnitTest::_check()",message);
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class UniqueArray2UnitTest
: public TraceAccessor
{
 public:
  UniqueArray2UnitTest(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

  void _printArray(const UniqueArray2<DataType>& array)
    {
      OStringStream ostr;
      Integer n1 = array.dim1Size();
      Integer n2 = array.dim2Size();
      ostr() << " Dim1=" << n1
             << " Dim2=" << n2
             << "\n";
      for( Integer i=0; i<n1; ++i ){
        ostr() << " I=[" << i << "] ->";
        for( Integer j=0; j<n2; ++j )
          ostr() << " [" << j << "]=" << array[i][j];
        ostr() << "\n";
      }
      ostr() << "\n";
      info() << ostr.str();
    }
  void _setArray(UniqueArray2<DataType>& array)
    {
      Integer n1 = array.dim1Size();
      Integer n2 = array.dim2Size();
      for( Integer i=0; i<n1; ++i )
        for( Integer j=0; j<n2; ++j )
          array[i][j] = (double)((i*n2)+j);
    }
  void _checkArray(UniqueArray2<DataType>& array,Integer original_dim2,Integer dim1,Integer dim2)
    {
      double total = DataType();
      for( Integer i=0; i<dim1; ++i )
        for( Integer j=0; j<dim2; ++j ){
          double v = (i*original_dim2)+j;
          info() << "VALUE i=" << i << " j=" << j << " v=" << v << " a=" << array[i][j];
          _check(array[i][j]==DataType(v),"Bad value");
          total += array[i][j];
        }
      info() << "TOTAL = " << total;
    }

  void _testSwap(bool use_own_swap)
  {
    // Test std::move() via std::swap() (si use_own_swap==false)
    // ou UniqueArray2::swap() (si use_own_swap==true)

    // Normalement les pointeurs des 2 tableaux doivent juste être échangés.
    UniqueArray2<DataType> c1(7,5);
    DataType* x1 = c1.viewAsArray().data();
    info() << "** C1_this = " << &c1;
    info() << "** C1_BASE = " << x1;
    UniqueArray2<DataType> c2(3,4);
    DataType* x2 = c2.viewAsArray().data();
    info() << "** C2_this = " << &c2;
    info() << "** C2_BASE = " << x2;

    if (use_own_swap)
      swap(c1,c2);
    else
      std::swap(c1,c2);

    DataType* after_x1 = c1.viewAsArray().data();
    DataType* after_x2 = c2.viewAsArray().data();
    info() << "** C1_BASE_AFTER = " << after_x1 << " size=" << c1.dim1Size();
    info() << "** C2_BASE_AFTER = " << after_x2 << " size=" << c2.dim1Size();

    _check(x1==after_x2,"Bad value after swap [1]");
    _check(x2==after_x1,"Bad value after swap [2]");
    _check(c1.dim1Size()==3,"Bad value after swap [3]");
    _check(c2.dim1Size()==7,"Bad value after swap [4]");
  }
  void _doTestWithAllocator(IMemoryAllocator* allocator)
  {
    info() << "Test with allocator v=" << allocator;

    UniqueArray2<DataType> array(allocator);
    array.resize(5,3);
    _check(array.dim1Size()==5,"Bad size");
    _check(array.dim2Size()==3,"Bad dim2size");
    _setArray(array);
    _checkArray(array,3,5,3);

    // Teste reduction de taille
    array.resize(7,2);
    _check(array.dim1Size()==7,"Bad size (2)");
    _check(array.dim2Size()==2,"Bad dim2size (1)");
    _checkArray(array,3,5,2);
    info() << " NB NEW=" << Wrapper<Real>::nb_new;

    // Teste augmentation de taille
    array.resize(9,5);
    _check(array.dim1Size()==9,"Bad size (2)");
    _check(array.dim2Size()==5,"Bad dim2size (1)");
    _checkArray(array,3,5,2);
    _printArray(array);
    info() << " NB NEW=" << Wrapper<Real>::nb_new;

    // Teste reduction de taille de la premiere dimension
    array.resize(7,5);
    _check(array.dim1Size()==7,"Bad size (2)");
    _check(array.dim2Size()==5,"Bad dim2size (1)");
    _checkArray(array,3,5,2);
    info() << " NB NEW=" << Wrapper<Real>::nb_new;
  }

  void doTest()
  {
    info() << "SIZEOF(Array)=" << sizeof(Array<DataType>);
    info() << "SIZEOF(UniqueArray)=" << sizeof(UniqueArray<DataType>);
    info() << "SIZEOF(SharedArray)=" << sizeof(SharedArray<DataType>);
    info() << "SIZEOF(UniqueArray2)=" << sizeof(UniqueArray2<DataType>);
    info() << "SIZEOF(SharedArray2)=" << sizeof(SharedArray2<DataType>);
    {
      UniqueArray2<DataType> array;
      info() << array.dim2Size();
      array.resize(5);
    }
    {
      UniqueArray2<DataType> array;
      info() << array.dim1Size();
      info() << array.dim2Size();
      array.resize(5,2);
      _check(array.dim1Size()==5,"Bad dim1size");
      _check(array.dim2Size()==2,"Bad dim2size");
      _printArray(array);
    }
    {
      UniqueArray2<DataType> array(5,2);
      info() << array.dim1Size();
      info() << array.dim2Size();
      _check(array.dim1Size()==5,"Bad dim1size");
      _check(array.dim2Size()==2,"Bad dim2size");
      _printArray(array);
    }
    {
      PrintableMemoryAllocator print_alloc;
      _doTestWithAllocator(nullptr);
      _doTestWithAllocator(&print_alloc);
    }
    {
      UniqueArray2<DataType> array;
      info() << array.dim2Size();
      array.resize(0,5);
      _check(array.dim1Size()==0,"Bad size (3)");
      _check(array.dim2Size()==5,"Bad dim2size (3)");
      array.resize(3,5);
      _printArray(array);
      array.add(DataType(6.3));
      _printArray(array);
      for( Integer j=0; j<5; ++j ){
        info() << "j=" << j << " v=" << array[3][j];
        _check(array[3][j]==ARCANE_REAL(6.3),"Bad value (4)");
      }
    }

    // Test clone
    {
      UniqueArray2<DataType> array;
      array.resize(3,5);
      _fill(array);
      {
        UniqueArray2<DataType> cloned_array(array.clone());
        _checkSame(array,cloned_array);
      }
      {
        UniqueArray2<DataType> implicit_cloned_array(array);
        _checkSame(array,implicit_cloned_array);
        Real old_value = implicit_cloned_array[2][3];
        implicit_cloned_array[2][3] = 1.0;
        _check(array[2][3]==old_value,"Bad implicit clone");
        const void* r1 = array.viewAsArray().unguardedBasePointer();
        const void* r2 = implicit_cloned_array.viewAsArray().unguardedBasePointer();
        _check(r1!=r2,"Bad same pointer");          
      }
      array.clear();
      info() << "ARRAY SIZE=" << array.dim1Size() << " " << array.dim2Size();
      UniqueArray2<DataType> new_array;
      info() << "NEW ARRAY SIZE=" << new_array.dim1Size() << " " << new_array.dim2Size();
      _check(array.dim1Size()==0,"Bad size1 (test clone)");
      _check(array.dim2Size()==0,"Bad size2 (test clone)");
    }

    // Test copy
    {
      UniqueArray2<DataType> array;
      array.resize(3,5);
      _fill(array);
      {
        UniqueArray2<DataType> copy_array;
        copy_array.copy(array);
        _checkSame(array,copy_array);
      }
    }

    // Teste les échanges.
    {
      _testSwap(false);
      _testSwap(true);
    }

    // Test std::move() avec la même origine et destination
    {
      UniqueArray2<DataType> c1(7,5);
      DataType* x1 = c1.viewAsArray().data();
      info() << "** C1_this = " << &c1 << "\n";
      c1 = std::move(c1);
      DataType* after_x1 = c1.viewAsArray().data();
      info() << "** C1_BASE_AFTER = " << after_x1
             << " dim1size=" << c1.dim1Size()
             << " dim2size=" << c1.dim2Size();
      _check(x1==after_x1,"Bad value after same std::move() [1]");
      // Dump le tableau pour vérifier que les adresses sont valides.
      dumpArray(std::cout,c1.viewAsArray().constView(),0);
    }
  }
  void _fill(Array2<DataType>& array)
  {
    Integer dim1_size = array.dim1Size();
    Integer dim2_size = array.dim2Size();
    for( Integer i=0; i<dim1_size; ++i ){
      for( Integer j=0; j<dim2_size; ++j ){
        array[i][j] = (DataType)(i*j);
      }
    }
  }
  void _checkSame(Array2<DataType>& array1,Array2<DataType>& array2)
  {
    Integer dim1_size = array1.dim1Size();
    Integer dim2_size = array1.dim2Size();
    Integer dim1_size_2 = array2.dim1Size();
    Integer dim2_size_2 = array2.dim2Size();
    _check(dim1_size==dim1_size_2,"Bad same size");
    _check(dim2_size==dim2_size_2,"Bad same size");
    for( Integer i=0; i<dim1_size; ++i ){
      for( Integer j=0; j<dim2_size; ++j ){
        _check(array1[i][j]==array2[i][j],"Bad value");
      }
    }
  }
  void _check(bool expression,const String& message)
    {
      if (!expression)
        throw FatalErrorException(A_FUNCINFO,message);
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class SharedArray2UnitTest
: public TraceAccessor
{
 public:
  SharedArray2UnitTest(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

  void _printArray(const SharedArray2<DataType>& array)
    {
      OStringStream ostr;
      Integer n1 = array.dim1Size();
      Integer n2 = array.dim2Size();
      ostr() << " Dim1=" << n1
             << " Dim2=" << n2
             << "\n";
      for( Integer i=0; i<n1; ++i ){
        ostr() << " I=[" << i << "] ->";
        for( Integer j=0; j<n2; ++j )
          ostr() << " [" << j << "]=" << array[i][j];
        ostr() << "\n";
      }
      ostr() << "\n";
      info() << ostr.str();
    }
  void _setArray(SharedArray2<DataType>& array)
    {
      Integer n1 = array.dim1Size();
      Integer n2 = array.dim2Size();
      for( Integer i=0; i<n1; ++i )
        for( Integer j=0; j<n2; ++j )
          array[i][j] = (double)((i*n2)+j);
    }
  void _checkArray(SharedArray2<DataType>& array,Integer original_dim2,Integer dim1,Integer dim2)
    {
      double total = DataType();
      for( Integer i=0; i<dim1; ++i )
        for( Integer j=0; j<dim2; ++j ){
          double v = (i*original_dim2)+j;
          info() << "VALUE i=" << i << " j=" << j << " v=" << v << " a=" << array[i][j];
          _check(array[i][j]==DataType(v),"Bad value");
          total += array[i][j];
        }
      info() << "TOTAL = " << total;
    }
  void doTest()
  {
    {
      SharedArray2<DataType> array;
      info() << array.dim2Size();
      array.resize(5);
    }
    {
      SharedArray2<DataType> array;
      info() << array.dim1Size();
      info() << array.dim2Size();
      array.resize(5,2);
      _check(array.dim1Size()==5,"Bad dim1size");
      _check(array.dim2Size()==2,"Bad dim2size");
      _printArray(array);
    }
    {
      SharedArray2<DataType> array(5,2);
      info() << array.dim1Size();
      info() << array.dim2Size();
      _check(array.dim1Size()==5,"Bad dim1size");
      _check(array.dim2Size()==2,"Bad dim2size");
      _printArray(array);
    }
    {
      SharedArray2<DataType> array;
      array.resize(5,3);
      _check(array.dim1Size()==5,"Bad size");
      _check(array.dim2Size()==3,"Bad dim2size");
      _setArray(array);
      _checkArray(array,3,5,3);
      
      // Teste reduction de taille
      array.resize(7,2);
      _check(array.dim1Size()==7,"Bad size (2)");
      _check(array.dim2Size()==2,"Bad dim2size (1)");
      _checkArray(array,3,5,2);
      info() << " NB NEW=" << Wrapper<Real>::nb_new;

      // Teste augmentation de taille
      array.resize(9,5);
      _check(array.dim1Size()==9,"Bad size (2)");
      _check(array.dim2Size()==5,"Bad dim2size (1)");
      _checkArray(array,3,5,2);
      _printArray(array);
      info() << " NB NEW=" << Wrapper<Real>::nb_new;

      // Teste reduction de taille de la premiere dimension
      array.resize(7,5);
      _check(array.dim1Size()==7,"Bad size (2)");
      _check(array.dim2Size()==5,"Bad dim2size (1)");
      _checkArray(array,3,5,2);
      info() << " NB NEW=" << Wrapper<Real>::nb_new;
    }

    {
      SharedArray2<DataType> array;
      info() << array.dim2Size();
      array.resize(0,5);
      _check(array.dim1Size()==0,"Bad size (3)");
      _check(array.dim2Size()==5,"Bad dim2size (3)");
      array.resize(3,5);
      _printArray(array);
      array.add(DataType(6.3));
      _printArray(array);
      for( Integer j=0; j<5; ++j ){
        info() << "j=" << j << " v=" << array[3][j];
        _check(array[3][j]==ARCANE_REAL(6.3),"Bad value (4)");
      }
    }

    // Test clone
    {
      SharedArray2<DataType> array;
      array.resize(3,5);
      _fill(array);
      {
        SharedArray2<DataType> cloned_array(array.clone());
        _checkSame(array,cloned_array);
      }
      array = SharedArray2<DataType>();
      info() << "ARRAY SIZE=" << array.dim1Size() << " " << array.dim2Size();
      SharedArray2<DataType> new_array;
      info() << "NEW ARRAY SIZE=" << new_array.dim1Size() << " " << new_array.dim2Size();
      _check(array.dim1Size()==0,"Bad size (test clone)");
    }

    // Test shared
    {
      SharedArray2<DataType> array;
      array.resize(3,5);
      _fill(array);

      SharedArray2<DataType> new_array(array);
      _checkSame(new_array,array);

      // Les tableaux sont partages et doivent avoir le meme pointeur de base.
      const void* r1 = array.viewAsArray().unguardedBasePointer();
      const void* r2 = new_array.viewAsArray().unguardedBasePointer();
      _check(r1==r2,"Bad same pointer");          
    }

    // Test copy
    {
      SharedArray2<DataType> array;
      array.resize(3,5);
      _fill(array);
      {
        SharedArray2<DataType> copy_array;
        copy_array.copy(array);
        _checkSame(array,copy_array);
      }
    }
  }
  void _fill(SharedArray2<DataType>& array)
  {
    Integer dim1_size = array.dim1Size();
    Integer dim2_size = array.dim2Size();
    for( Integer i=0; i<dim1_size; ++i ){
      for( Integer j=0; j<dim2_size; ++j ){
        array[i][j] = (DataType)(i*j);
      }
    }
  }
  void _checkSame(SharedArray2<DataType>& array1,SharedArray2<DataType>& array2)
  {
    Integer dim1_size = array1.dim1Size();
    Integer dim2_size = array1.dim2Size();
    Integer dim1_size_2 = array2.dim1Size();
    Integer dim2_size_2 = array2.dim2Size();
    _check(dim1_size==dim1_size_2,"Bad same size");
    _check(dim2_size==dim2_size_2,"Bad same size");
    for( Integer i=0; i<dim1_size; ++i ){
      for( Integer j=0; j<dim2_size; ++j ){
        _check(array1[i][j]==array2[i][j],"Bad value");
      }
    }
  }
  void _check(bool expression,const String& message)
    {
      if (!expression)
        throw FatalErrorException("Array2UnitTest::_check()",message);
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class MultiArray2UnitTest
: public TraceAccessor
{
 public:
  MultiArray2UnitTest(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

  void _fill(MultiArray2<DataType>& array)
  {
    Integer dim1_size = array.dim1Size();
    for( Integer i=0; i<dim1_size; ++i ){
      ArrayView<DataType> a = array[i];
      Integer dim2_size = a.size();
      for( Integer j=0; j<dim2_size; ++j ){
        array[i][j] = (DataType)(i*j);
      }
    }
  }

  void _printArray(const MultiArray2<DataType>& array)
    {
      OStringStream ostr;
      Integer n1 = array.dim1Size();
      ostr() << " MultiArray2 Total=" << array.totalNbElement() << " Dim1=" << n1
             << "\n";
      for( Integer i=0; i<n1; ++i ){
        ostr() << " I=[" << i << "] ->";
        ConstArrayView<DataType> sub_array = array[i];
        Integer n2 = sub_array.size();
        for( Integer j=0; j<n2; ++j )
          ostr() << " [" << j << "]=" << array[i][j];
        ostr() << "\n";
      }
      ostr() << "\n";
      info() << ostr.str();
    }
  void _setArray(MultiArray2<DataType>& array)
    {
      Integer n1 = array.dim1Size();
      for( Integer i=0; i<n1; ++i ){
        Integer n2 = array[i].size();
        for( Integer j=0; j<n2; ++j )
          array[i][j] = (double)((i*n2)+j);
      }
    }
#if 0
  void _CheckArray(Array2<DataType>& array,Integer original_dim2,Integer dim1,Integer dim2)
    {
      double total = DataType();
      for( Integer i=0; i<dim1; ++i )
        for( Integer j=0; j<dim2; ++j ){
          double v = (i*original_dim2)+j;
          info() << "VALUE i=" << i << " j=" << j << " v=" << v << " a=" << array[i][j];
          _check(array[i][j]==v,"Bad value");
          total += array[i][j];
        }
      info() << "TOTAL = " << total;
    }
#endif

  void _checkSame(MultiArray2<DataType>& array1,MultiArray2<DataType>& array2)
  {
    Integer dim1_size = array1.dim1Size();
    Integer dim2_size = array2.dim1Size();
    _check(dim1_size==dim2_size,"Bad same size");
    for( Integer i=0; i<dim1_size; ++i ){
      ArrayView<DataType> a1 = array1[i];
      ArrayView<DataType> a2 = array2[i];

      Integer dim2_size1 = a1.size();
      Integer dim2_size2 = a2.size();
      _check(dim2_size1==dim2_size2,"Bad dim2 size");

      for( Integer j=0; j<dim2_size1; ++j ){
        _check(array1[i][j]==array2[i][j],"Bad value");
      }
    }
  }

  void _checkSize(ConstMultiArray2View<DataType> array,IntegerConstArrayView expected_sizes)
  {
    IntegerConstArrayView dim2_sizes = array.dim2Sizes();
    Integer array_size = dim2_sizes.size();
    Integer size = expected_sizes.size();
    _check(array_size==size,"Bad size");
    for( Integer i=0; i<size; ++i ){
      _check(dim2_sizes[i]==expected_sizes[i],"Bad dim2 size");
    }
  }

  void doTest()
  {
    {
      MultiArray2<DataType> array;

      {
        IntegerUniqueArray sizes(5);
        for( Integer i=0, is=sizes.size(); i<is; ++i )
          sizes[i] = (5+i) % 5;
        array.resize(sizes);

        _printArray(array);
        _checkSize(array,sizes);
        _setArray(array);
        _printArray(array);

        // Vérifie une augmentation de la taille
        sizes.resize(9);
        for( Integer i=0, is=sizes.size(); i<is; ++i )
          sizes[i] = (2+i) % 4;
        array.resize(sizes);
        _checkSize(array,sizes);
        _printArray(array);
        _setArray(array);
        _printArray(array);

        // Vérifie une diminution de la taille
        sizes.resize(7);
        for( Integer i=0, is=sizes.size(); i<is; ++i )
          sizes[i] = (i) % 5;
        array.resize(sizes);
        _checkSize(array,sizes);
        _printArray(array);

        // Vérifie un resize ne changeant pas la taille
        array.resize(sizes);
        _checkSize(array,sizes);
        _printArray(array);
      }
      
      // Test clone
      {
        UniqueArray<Integer> sizes(5);
        for( Integer i=0, n=sizes.size(); i<n; ++i )
          sizes[i] = i+2;
        SharedMultiArray2<DataType> array(sizes);
        _fill(array);
        {
          SharedMultiArray2<DataType> cloned_array(array.clone());
          _checkSame(array,cloned_array);
          // Modifie le clone et vérifie que l'original n'a pas changé.
          cloned_array[2][3] = 0;
          _check(array[2][3]!=0,"Bad value (test clone)");
        }
        array = SharedMultiArray2<DataType>();
        SharedMultiArray2<DataType> new_array;
        _check(array.dim1Size()==0,"Bad size (test clone)");
      }

      // Test shared
      {
        UniqueArray<Integer> sizes(5);
        for( Integer i=0, n=sizes.size(); i<n; ++i )
          sizes[i] = i+2;
        SharedMultiArray2<DataType> array(sizes);
        _fill(array);

        SharedMultiArray2<DataType> new_array(array);
        _checkSame(new_array,array);
        {
          // Modifie la nouvelle référence et vérifie que l'original a changé.
          new_array[2][3] = 0.0;
          _check(array[2][3]==0.0,"Bad value (test shared)");
        }

        // Les tableaux sont partages et doivent avoir le meme pointeur de base.
        const void* r1 = array.viewAsArray().unguardedBasePointer();
        const void* r2 = new_array.viewAsArray().unguardedBasePointer();
        _check(r1==r2,"Bad same pointer");
      }

      // Test copie à partir d'un SharedArray
      {
        UniqueArray<Integer> sizes(5);
        for( Integer i=0, n=sizes.size(); i<n; ++i )
          sizes[i] = i+2;
        SharedMultiArray2<DataType> array(sizes);
        _fill(array);

        {
          UniqueMultiArray2<DataType> new_array(array);
          _checkSame(new_array,array);

          // Les tableaux ne sont pas partagés et ne doivent avoir le meme pointeur de base.
          const void* r1 = array.viewAsArray().unguardedBasePointer();
          const void* r2 = new_array.viewAsArray().unguardedBasePointer();
          _check(r1!=r2,"Bad same pointer for unique array");

          {
            // Modifie la nouvelle référence et vérifie que l'original n'a pas changé.
            new_array[2][3] = 0.0;
            _check(array[2][3]!=0.0,"Bad value (test shared with unique array)");
          }
        }
        {
          UniqueMultiArray2<DataType> new_array(array);
          _checkSame(new_array,array);
          new_array[2][3] = 0.0;
          UniqueMultiArray2<DataType> new2_array(new_array);
          _checkSame(new2_array,new_array);
          new2_array[2][3] = -5.0;
          {
            // Modifie la nouvelle référence et vérifie que l'original n'a pas changé.
            _check(new_array[2][3]==0.0,"Bad value (test clone unique array)");
          }
        }
      }

      {
        // Test accès en mode check lorsque certaines tailles sont nulles.
        IntegerUniqueArray d;
        UniqueMultiArray2<Real> ma;
        Integer n = 10;
        d.resize(n);
        for( Integer i=0; i<n; ++i ){
          d[i] = i % 3;
        }
        ma.resize(d);
        const UniqueMultiArray2<Real>& const_ma = ma;
        for( Integer i=0; i<n; ++i ){
          ma[i].fill(0.0);
          info() << "S=" << const_ma[i].size();
        }
      }

    }
  }
  void _check(bool expression,const String& message)
  {
    if (!expression)
      throw FatalErrorException("MutliArrayUnitTest::_check()",message);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArrayUnitTestNamespace
{
class Index
{
 public:
  void createFromInteger(Integer i,Integer mid_size)
  {
    m_array_index = i / mid_size;
    m_value_index = i - (m_array_index*mid_size);
  };

 public:
  Int32 m_array_index;
  Int32 m_value_index;
};

template<typename DataType>
class ArrayList
{
 public:

  void resize(Integer size)
  {
    m_array.resize(size);
    m_views[0] = m_array.subView(0,size/2);
    m_views[1] = m_array.subView(size/2,size/2);
  }

 public:
  const DataType operator[](Integer index) const
  {
    //return m_views[ index & 0x1 ][ index >> 1 ];
    return m_views[ (index & (1<<30)) >> 30 ][ index & ~(1<<30) ];
  }
  const DataType operator[](Index index) const
  {
    return m_views[ index.m_array_index ][ index.m_value_index ];
  }
  void setValue(Integer index,const DataType& value)
  {
    m_views[ (index & (1<<30)) >> 30 ][ index & ~(1<<30) ] = value;
  }
  void setValue(Index index,const DataType& value)
  {
    m_views[ index.m_array_index ][ index.m_value_index ] = value;
  }
  ArrayView<DataType> m_views[2];
  UniqueArray<DataType> m_array;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du maillage
 */
class ArrayUnitTest
: public BasicUnitTest
{
  class BasicStruct
  {
   public:
    BasicStruct(int v) : m_value(v){}
    int m_value;
  };

 public:

  ArrayUnitTest(const ServiceBuildInfo& cb);
  ~ArrayUnitTest();

 public:

  virtual void initializeTest() {}
  virtual void executeTest();

 private:

  Real f1();
  Real f2();
  Real f2_1();
  Real f3();
  Real f3_1();
  Real f3_2();
  void f4();
  void f5();
  void f6();
  void f7();
  Real f7_1();
  Real f8();
  Real f8_1();
  Real f8_2();
  void _Add(RealArray& v,Integer new_size);
  void _TestArrayDim2();
  void _TestMultiArray2();
  void _TestPerfs();
  void _check(bool expression,const String& message);
  void _TestArray();
  void _TestInterval(Integer nb_interval);
  void _TestAddStruct();
  void _TestConstStruct();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(ArrayUnitTest,IUnitTest,ArrayUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayUnitTest::
ArrayUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayUnitTest::
~ArrayUnitTest()
{
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArrayTimer
{
  public:
  ArrayTimer(const char* msg)
  : m_begin_time(0.0), m_msg(msg)
  {
    m_begin_time = platform::getRealTime();
  }
  ~ArrayTimer()
  {
    Real end_time = platform::getRealTime();
    Real true_time_v = end_time - m_begin_time;
    double true_time = (double)(true_time_v);
    std::cout << " -- -- Time: ";
    std::cout.width(40);
    std::cout << m_msg << " = " << (true_time) << '\n';
  }
  Real m_begin_time;
  const char* m_msg;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
executeTest()
{
#if 1
  _TestArray();
  _TestArrayDim2();
  _TestMultiArray2();
  _TestAddStruct();
  _TestConstStruct();
#endif
#if 0
  _TestPerfs();
  f1();
  f2();
  info() << "F2_1 " << f2_1();
  info() << "F3=  " << f3();
  info() << "F3_1=" << f3_1();
  info() << "F3_2=" << f3_2();
  f4();
  f5();
  f6();
  f7();
  info() << "F7_1=" << f7_1();
  info() << "F8=" << f8();
  info() << "F8_1=" << f8_1();
  info() << "F8_2=" << f8_2();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const int NB_Z = 200;
static const int SIZE = 15000*10;
const int TRUE_SIZE = SIZE*3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView _viewInterval(Int64Array& array,Integer index,Integer nb_interval)
{
  Integer n = array.size();
  Integer isize = n / nb_interval;
  Integer ibegin = index * isize;
  // Pour le dernier interval, prend les elements restants
  if ((index+1)==nb_interval)
    isize = n - ibegin;
  return Int64ConstArrayView(isize,array.data()+ibegin);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
_TestArray()
{
  for( Integer i=1; i<13; ++i )
    _TestInterval(i);

  RealUniqueArray v(1200);
  for( Integer i=0, n=v.size(); i<n; ++i ){
    Real a = (Real)i;
    v[i] = (a*a + 2.0);
  }
  info() << " ValueSize1= " << v.size() << " values=" << v;
  v.resize(240);
  info() << " ValueSize2= " << v.size() << " values=" << v;
  OStringStream ostr;
  dumpArray(ostr(),v.constView(),100);
  info() << " ValueSize3= " << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
_TestInterval(Integer nb_interval)
{
  for( Integer n=0; n<100; ++n ){
    UniqueArray<Int64> array(n);
    Int64 my_total = 0;
    for( Integer i=0; i<n; ++i ){
      array[i] = (i+1);
      my_total += (i+1);
    }
    Int64 interval_total = 0;
    for( Integer i=0; i<nb_interval; ++i ){
      Int64ConstArrayView v = array.view().subViewInterval(i,nb_interval);
      for( Integer z=0, zs=v.size(); z<zs; ++z )
        interval_total += v[z];
    }
    if (interval_total!=my_total)
      fatal() << "Bad total expected=" << my_total << " found=" << interval_total;
    info() << "Size=" << n << " interval=" << nb_interval << " total=" << my_total;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef UniqueArray<Real> RealUniqueArray;
typedef ArrayFullAccessorT<Real> RealArrayFullAccessor;

typedef UniqueArray<Real3> Real3UniqueArray;
typedef ArrayFullAccessorT<Real3> Real3ArrayFullAccessor;

void ArrayUnitTest::
_Add(RealArray& v,Integer new_size)
{
  v.resize(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f1()
{
  ArrayTimer tm("f1 vector,direct,Real");
  std::vector<Real> a, b, c, d, e;
  a.resize(TRUE_SIZE);
  b.resize(TRUE_SIZE);
  c.resize(TRUE_SIZE);
  d.resize(TRUE_SIZE);
  e.resize(TRUE_SIZE);
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Real z = (Real)i;
    b[i] = c[i] = d[i] = e[i] = z;
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
    s += a[z%5];
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f2()
{
  ArrayTimer tm("f2 direct,Real");
  RealUniqueArray v1, v2, v3, v4, v5;
  RealArrayFullAccessor a(v1), b(v2), c(v3), d(v4), e(v5);
  a.resize(TRUE_SIZE);
  b.resize(TRUE_SIZE);
  c.resize(TRUE_SIZE);
  d.resize(TRUE_SIZE);
  e.resize(TRUE_SIZE);
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Real z = (Real)i;
    b[i] = c[i] = d[i] = e[i] = z;
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
    s += a[z%5];
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f2_1()
{
  ArrayTimer tm("f2_1 direct,Real,noaccessor");
  RealUniqueArray a, b, c, d, e;
  a.resize(TRUE_SIZE);
  b.resize(TRUE_SIZE);
  c.resize(TRUE_SIZE);
  d.resize(TRUE_SIZE);
  e.resize(TRUE_SIZE);
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Real z = (Real)i;
    a[i] = b[i] = c[i] = d[i] = e[i] = z;
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
    s += a[z%5];
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f3()
{
  ArrayTimer tm("f3 direct,Real,ivdep");
  RealUniqueArray v1, v2, v3, v4, v5;
  RealArrayFullAccessor a(v1), b(v2), c(v3), d(v4), e(v5);
  a.resize(TRUE_SIZE);
  b.resize(TRUE_SIZE);
  c.resize(TRUE_SIZE);
  d.resize(TRUE_SIZE);
  e.resize(TRUE_SIZE);
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Real z = (Real)i;
    a[i] = b[i] = c[i] = d[i] = e[i] = z;
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
#ifdef __ia64
#pragma ivdep
#endif
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
    s += a[z%5];
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f3_1()
{
  ArrayTimer tm("f3_1 direct,Real,ivdep,noaccessor");
  RealUniqueArray a, b, c, d, e;
  a.resize(TRUE_SIZE);
  b.resize(TRUE_SIZE);
  c.resize(TRUE_SIZE);
  d.resize(TRUE_SIZE);
  e.resize(TRUE_SIZE);
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Real z = (Real)i;
    a[i] = b[i] = c[i] = d[i] = e[i] = z;
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){

#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
    s += a[z%5];
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f3_2()
{
  ArrayTimer tm("f3_2 direct,ptr,Real,ivdep,noaccessor");
  Real *a = new Real[TRUE_SIZE];
  Real *b = new Real[TRUE_SIZE];
  Real *c = new Real[TRUE_SIZE];
  Real *d = new Real[TRUE_SIZE];
  Real *e = new Real[TRUE_SIZE];
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Real z = (Real)i;
    a[i] = b[i] = c[i] = d[i] = e[i] = z;
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
#pragma ivdep
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
    s += a[z%5];
  }
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
f4()
{
  ArrayTimer tm("f4 direct,Real3");
  Real3UniqueArray v1, v2, v3, v4, v5;
  Real3ArrayFullAccessor a(v1), b(v2), c(v3), d(v4), e(v5);
  a.resize(SIZE);
  b.resize(SIZE);
  c.resize(SIZE);
  d.resize(SIZE);
  e.resize(SIZE);
  for( Integer i=0, is=SIZE; i<is; ++i ){
    Real z = (Real)i;
    a[i] = b[i] = d[i] = e[i] = Real3(z,z,z);
  }
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=SIZE; i<is; ++i ){
      v1[i] = v2[i] + v3[i] * v4[i] + v5[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
f5()
{
  ArrayTimer tm("f5 direct,Real3,ivdep");
  Real3UniqueArray v1, v2, v3, v4, v5;
  Real3ArrayFullAccessor a(v1), b(v2), c(v3), d(v4), e(v5);
  a.resize(SIZE);
  b.resize(SIZE);
  c.resize(SIZE);
  d.resize(SIZE);
  e.resize(SIZE);
  for( Integer i=0, is=SIZE; i<is; ++i ){
    Real z = (Real)i;
    a[i] = b[i] = d[i] = e[i] = Real3(z,z,z);
  }
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){

#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for( Integer i=0, is=SIZE; i<is; ++i ){
      v1[i] = v2[i] + v3[i] * v4[i] + v5[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
f6()
{
  ArrayTimer tm("f6 indirect,Real,ivdep");
  RealUniqueArray v1, v2, v3, v4, v5;
  RealArrayFullAccessor a(v1), b(v2), c(v3), d(v4), e(v5);
  a.resize(TRUE_SIZE);
  b.resize(TRUE_SIZE);
  c.resize(TRUE_SIZE);
  d.resize(TRUE_SIZE);
  e.resize(TRUE_SIZE);
  Integer* index = new Integer[TRUE_SIZE];
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    index[i] = i;
    Real z = (Real)i;
    a[i] = b[i] = d[i] = e[i] = z;
  }
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      v1[index[i]] = v2[index[i]] + v3[index[i]] * v4[index[i]] + v5[index[i]];
    }
  }
  delete[] index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
f7()
{
  ArrayTimer tm("f7 indirect,Real3,ivdep");
  Real3UniqueArray v1, v2, v3, v4, v5;
  Real3ArrayFullAccessor a(v1), b(v2), c(v3), d(v4), e(v5);
  a.resize(SIZE);
  b.resize(SIZE);
  c.resize(SIZE);
  d.resize(SIZE);
  e.resize(SIZE);
  Integer* index = new Integer[SIZE];
  for( Integer i=0, is=SIZE; i<is; ++i ){
    index[i] = i;
    Real z = (Real)i;
    a[i] = b[i] = d[i] = e[i] = Real3(z,z,z);
  }
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for( Integer i=0, is=SIZE; i<is; ++i ){
      v1[index[i]] = v2[index[i]] + v3[index[i]] * v4[index[i]] + v5[index[i]];
    }
  }
  delete[] index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f7_1()
{
  ArrayTimer tm("f7_1 indirect,Real3");
  Real3UniqueArray v1, v2, v3, v4, v5;
  Real3ArrayFullAccessor a(v1), b(v2), c(v3), d(v4), e(v5);
  a.resize(SIZE);
  b.resize(SIZE);
  c.resize(SIZE);
  d.resize(SIZE);
  e.resize(SIZE);
  Integer* index = new Integer[SIZE];
  for( Integer i=0, is=SIZE; i<is; ++i ){
    index[i] = i;
    Real z = (Real)i;
    a[i] = b[i] = d[i] = e[i] = Real3(z,z,z);
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=SIZE; i<is; ++i ){
      v1[index[i]] = v2[index[i]] + v3[index[i]] * v4[index[i]] + v5[index[i]];
    }
    s += v1[z%5].x;
  }
  delete[] index;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f8()
{
  using namespace ArrayUnitTestNamespace;
  ArrayTimer tm("f8 indirect,ArrayList,Real3");
  ArrayList<Real3> a, b, c, d, e;
  a.resize(SIZE);
  b.resize(SIZE);
  c.resize(SIZE);
  d.resize(SIZE);
  e.resize(SIZE);
  Integer* index = new Integer[SIZE];
  for( Integer i=0, is=SIZE; i<is; ++i ){
    index[i] = i;
    Real z = (Real)i;
    Real3 z3 = Real3(z,z,z);
    a.setValue(i,z3);
    b.setValue(i,z3);
    c.setValue(i,z3);
    d.setValue(i,z3);
    e.setValue(i,z3);
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=SIZE; i<is; ++i ){
      a.setValue(index[i], b[index[i]] + c[index[i]] * d[index[i]] + e[index[i]]);
    }
    s += a[z%5].x;
  }
  delete[] index;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f8_1()
{
  using namespace ArrayUnitTestNamespace;
  ArrayTimer tm("f8_1 direct,ArrayList,Real3");
  ArrayList<Real3> a, b, c, d, e;
  a.resize(SIZE);
  b.resize(SIZE);
  c.resize(SIZE);
  d.resize(SIZE);
  e.resize(SIZE);
  for( Integer i=0, is=SIZE; i<is; ++i ){
    Real z = (Real)i;
    Real3 z3 = Real3(z,z,z);
    a.setValue(i,z3);
    b.setValue(i,z3);
    c.setValue(i,z3);
    d.setValue(i,z3);
    e.setValue(i,z3);
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=SIZE; i<is; ++i ){
      a.setValue(i, b[i] + c[i] * d[i] + e[i]);
    }
    s += a[z%5].x;
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ArrayUnitTest::
f8_2()
{
  using namespace ArrayUnitTestNamespace;
  ArrayTimer tm("f8 indirect(Index),ArrayList,Real3");
  ArrayList<Real3> a, b, c, d, e;
  a.resize(SIZE);
  b.resize(SIZE);
  c.resize(SIZE);
  d.resize(SIZE);
  e.resize(SIZE);
  Index* index = new Index[SIZE];
  for( Integer i=0, is=SIZE; i<is; ++i ){
    index[i].createFromInteger(i,SIZE/2);
    Real z = (Real)i;
    Real3 z3 = Real3(z,z,z);
    a.setValue(i,z3);
    b.setValue(i,z3);
    c.setValue(i,z3);
    d.setValue(i,z3);
    e.setValue(i,z3);
  }
  Real s = 0.0;
  for( Integer z=0, iz=NB_Z; z<iz; ++z ){
    for( Integer i=0, is=SIZE; i<is; ++i ){
      Index idx = index[i];
      a.setValue(idx, b[idx] + c[idx] * d[idx] + e[idx]);
    }
    s += a[z%5].x;
  }
  delete[] index;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
_TestArrayDim2()
{
  {
    Array2UnitTest<Real> a2(traceMng());
    a2.doTest();
  }
  info() << " NB NEW FIRST=" << Wrapper<Real>::nb_new;
  for( Integer i=0; i<5; ++i ){
    UniqueArray2UnitTest< Wrapper<Real> > a2(traceMng());
    a2.doTest();
    info() << " NB NEW=" << Wrapper<Real>::nb_new;
  }


  info() << "TESTING UniqueArray2";
  {
    UniqueArray2UnitTest<Real> a2(traceMng());
    a2.doTest();
  }

  info() << " NB NEW FIRST=" << Wrapper<Real>::nb_new;
  for( Integer i=0; i<5; ++i ){
    UniqueArray2UnitTest< Wrapper<Real> > a2(traceMng());
    a2.doTest();
    info() << " NB NEW=" << Wrapper<Real>::nb_new;
  }

  info() << "TESTING SharedArray2";
  {
    SharedArray2UnitTest<Real> a2(traceMng());
    a2.doTest();
  }
  info() << " NB NEW FIRST=" << Wrapper<Real>::nb_new;
  for( Integer i=0; i<5; ++i ){
    SharedArray2UnitTest< Wrapper<Real> > a2(traceMng());
    a2.doTest();
    info() << " NB NEW=" << Wrapper<Real>::nb_new;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void g2_0(MultiArray2<Int64>& array)
{
  for( Integer i=0, is=array.dim1Size(); i<is; ++i ){
    for( Integer j=0, js=array[i].size(); j<js; ++j )
      //array[i][j] = i*j;
      array.at(i,j) = i*j;
  }
}

void g2_2(MultiArray2<Int64>& array)
{
  for( Integer i=0, is=array.dim1Size(); i<is; ++i ){
    Int64ArrayView cav(array[i]);
    for( Integer j=0, js=cav.size(); j<js; ++j )
      cav[j] = i*j;
  }
}

void g2_6(Array2<Int64>& array)
{
  for( Integer i=0, is=array.dim1Size(); i<is; ++i ){
    Int64ArrayView cav(array[i]);
    for( Integer j=0, js=cav.size(); j<js; ++j )
      cav[j] = i*j;
  }
}

void ArrayUnitTest::
_TestMultiArray2()
{
  info() << " ** TEST MULTI ARRAY";
  {
    MultiArray2UnitTest<Real> a2(traceMng());
    a2.doTest();
  }
  info() << " NB NEW FIRST=" << Wrapper<Real>::nb_new;
  for( Integer i=0; i<5; ++i ){
    MultiArray2UnitTest< Wrapper<Real> > a2(traceMng());
    a2.doTest();
    info() << " NB NEW=" << Wrapper<Real>::nb_new;
  }
}

void ArrayUnitTest::
_TestPerfs()
{
  IApplication* app = subDomain()->application();
  ServiceFinder2T<IProfilingService,IApplication> sf(app,app);
  auto ps = sf.createReference("PapiProfilingService");
  ITimerMng* tm = 0;
  if (ps.get())
    tm = ps->timerMng();
  if (!tm)
    tm = subDomain()->timerMng();
  //return;
  //Timer timer(tm,"Test",Timer::TimerReal);
  Timer timer(subDomain(),"Test",Timer::TimerReal);
  {
    SharedMultiArray2<Int64> array;
    Integer n = 1000000;
    // Au moins 10 iterations pour supprimer les effets de cache
    Integer nb_iter = 3;
    IntegerUniqueArray sizes(n);
    sizes.fill(35);

    array.resize(sizes);

    UniqueArray2<Int64> array2;
    array2.resize(n,35);

    {
      Timer::Sentry ts(&timer);
      for( Integer z=0; z<nb_iter; ++ z){
        g2_0(array);
      }
    }
    info() << "TIME1 = " << timer.lastActivationTime();
    
    {
      Timer::Sentry ts(&timer);
      for( Integer z=0; z<nb_iter; ++ z){
        g2_2(array);
      }
    }
    info() << "TIME2 = " << timer.lastActivationTime();


    {
      Timer::Sentry ts(&timer);
      for( Integer z=0; z<nb_iter; ++ z){
        for( Integer i=0, is=array2.dim1Size(); i<is; ++i ){
          for( Integer j=0, js=array2[i].size(); j<js; ++j )
            array2[i][j] = i*j;
        }
      }
    }
    info() << "TIME3 = " << timer.lastActivationTime();

    {
      Timer::Sentry ts(&timer);
      for( Integer z=0; z<nb_iter; ++ z){
        for( Integer i=0, is=array2.dim1Size(); i<is; ++i ){
          for( Integer j=0, js=array2[i].size(); j<js; ++j )
            array2.setItem(i,j,i*j);
        }
      }
    }
    info() << "TIME4 = " << timer.lastActivationTime();

    {
      Timer::Sentry ts(&timer);
      for( Integer z=0; z<nb_iter; ++ z){
        for( Integer i=0, is=array2.dim1Size(); i<is; ++i ){
          for( Integer j=0, js=array2[i].size(); j<js; ++j )
            array[i][j] = i*j;
        }
      }
    }
    info() << "TIME5 = " << timer.lastActivationTime();

    {
      Timer::Sentry ts(&timer);
      for( Integer z=0; z<nb_iter; ++ z){
        g2_6(array2);
      }
    }
    info() << "TIME6 = " << timer.lastActivationTime();
  }
}

namespace
{
struct FoundInfo
{
  Int32 contrib_owner;
  Int32 owner;
  Int64 local_id;
  Int32 face_local_id;
  Int32 orig_face_local_id;
  Int32 user_value;
  Real distance;
  Real3POD intersection;
};
}

void ArrayUnitTest::
_TestAddStruct()
{
  info() << "Test Add Struct";

  SharedArray<FoundInfo> found_infos;
  Integer n = 250;

  SharedArray<FoundInfo> found_infos2(found_infos);
  for( Integer i=0; i<n; ++i ){
    FoundInfo fi;
    if ((i%3)==0)
      continue;
    fi.contrib_owner = 5;
    fi.owner = i / 2;
    fi.local_id = i;
    fi.face_local_id = i-3;
    fi.orig_face_local_id = i+3;
    fi.user_value = i;
    fi.intersection.x = 2.0;
    fi.intersection.y = 1.0;
    fi.intersection.z = 3.0;
    fi.distance = -5.2;

    found_infos.add(fi);
  }

  for( Integer i=0; i<found_infos.size(); ++i ){
    const FoundInfo& fi = found_infos[i];
    info() << "V=" << i << " owner=" << fi.contrib_owner << " local_id=" << fi.local_id
           << " user_value=" << fi.user_value << " distance=" << fi.distance;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUnitTest::
_TestConstStruct()
{
  info() << "Test Const Struct";

  UniqueArray<const BasicStruct*> values;
  const BasicStruct* bs = new BasicStruct(5);
  values.add(bs);
  info() << "V0=" << values[0]->m_value;
  delete bs;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
