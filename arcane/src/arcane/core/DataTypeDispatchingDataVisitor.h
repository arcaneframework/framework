// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypeDispatchingDataVisitor.h                            (C) 2000-2025 */
/*                                                                           */
/* IDataVisitor dispatching operations according to the data type.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATATYPEDISPATCHINGDATAVISITOR_H
#define ARCANE_CORE_DATATYPEDISPATCHINGDATAVISITOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/AbstractDataVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Template class for dispatching data (IData)
 * according to their type (DataType).
 */
template<typename DataType>
class ARCANE_CORE_EXPORT IDataTypeDataDispatcherT
{
 public:
  virtual ~IDataTypeDataDispatcherT(){}
 public:
  virtual void applyDispatch(IScalarDataT<DataType>* data) =0;
  virtual void applyDispatch(IArrayDataT<DataType>* data) =0;
  virtual void applyDispatch(IArray2DataT<DataType>* data) =0;
  virtual void applyDispatch(IMultiArray2DataT<DataType>*) {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Specialization of IDataDispatcherT for the 'String' class.
 *
 * This specialization is necessary because there is no
 * data of type \a IMultiArray2Data<String>.
 */
template<>
class ARCANE_CORE_EXPORT IDataTypeDataDispatcherT<String>
{
 public:
  virtual ~IDataTypeDataDispatcherT(){}
 public:
  virtual void applyDispatch(IScalarDataT<String>* data) =0;
  virtual void applyDispatch(IArrayDataT<String>* data) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief IDataVisitor dispatching operations according to the data type.
 *
 * The dispatcher must inherit from IDataDispatcherT and the interface
 * \a IDispatcherType passed as an argument to the template class.
 *
 * The \a IDispatcherType interface must define the type
 * HasStringDispatch as TrueType or FalseType depending on whether it
 * supports String type dispatch.
 *
 * It is possible to construct an instance directly via
 * the constructor or via the static function create().
 */
class ARCANE_CORE_EXPORT AbstractDataTypeDispatchingDataVisitor
: public AbstractDataVisitor
{
 public:

  /*!
   * \brief Constructs an instance.
   * The objects passed as parameters become the property of this
   * instance, which is responsible for destroying them via the operator
   * delete in the destructor
   */
  AbstractDataTypeDispatchingDataVisitor(IDataTypeDataDispatcherT<Byte>* a_byte,
                                         IDataTypeDataDispatcherT<Real>* a_real,
                                         IDataTypeDataDispatcherT<Int16>* a_int16,
                                         IDataTypeDataDispatcherT<Int32>* a_int32,
                                         IDataTypeDataDispatcherT<Int64>* a_int64,
                                         IDataTypeDataDispatcherT<Real2>* a_real2,
                                         IDataTypeDataDispatcherT<Real3>* a_real3,
                                         IDataTypeDataDispatcherT<Real2x2>* a_real2x2,
                                         IDataTypeDataDispatcherT<Real3x3>* a_real3x3,
                                         IDataTypeDataDispatcherT<String>* a_string
                                         );
  ~AbstractDataTypeDispatchingDataVisitor();

 public:

  void applyVisitor(IScalarDataT<Byte>* data) override;
  void applyVisitor(IScalarDataT<Real>* data) override;
  void applyVisitor(IScalarDataT<Int16>* data) override;
  void applyVisitor(IScalarDataT<Int32>* data) override;
  void applyVisitor(IScalarDataT<Int64>* data) override;
  void applyVisitor(IScalarDataT<Real2>* data) override;
  void applyVisitor(IScalarDataT<Real3>* data) override;
  void applyVisitor(IScalarDataT<Real2x2>* data) override;
  void applyVisitor(IScalarDataT<Real3x3>* data) override;
  void applyVisitor(IScalarDataT<String>* data) override;

  void applyVisitor(IArrayDataT<Byte>* data) override;
  void applyVisitor(IArrayDataT<Real>* data) override;
  void applyVisitor(IArrayDataT<Int16>* data) override;
  void applyVisitor(IArrayDataT<Int32>* data) override;
  void applyVisitor(IArrayDataT<Int64>* data) override;
  void applyVisitor(IArrayDataT<Real2>* data) override;
  void applyVisitor(IArrayDataT<Real3>* data) override;
  void applyVisitor(IArrayDataT<Real2x2>* data) override;
  void applyVisitor(IArrayDataT<Real3x3>* data) override;
  void applyVisitor(IArrayDataT<String>* data) override;

  void applyVisitor(IArray2DataT<Byte>* data) override;
  void applyVisitor(IArray2DataT<Real>* data) override;
  void applyVisitor(IArray2DataT<Int16>* data) override;
  void applyVisitor(IArray2DataT<Int32>* data) override;
  void applyVisitor(IArray2DataT<Int64>* data) override;
  void applyVisitor(IArray2DataT<Real2>* data) override;
  void applyVisitor(IArray2DataT<Real3>* data) override;
  void applyVisitor(IArray2DataT<Real2x2>* data) override;
  void applyVisitor(IArray2DataT<Real3x3>* data) override;

  void applyVisitor(IMultiArray2DataT<Byte>*) override {}
  void applyVisitor(IMultiArray2DataT<Real>*) override {}
  void applyVisitor(IMultiArray2DataT<Int16>*) override {}
  void applyVisitor(IMultiArray2DataT<Int32>*) override {}
  void applyVisitor(IMultiArray2DataT<Int64>*) override {}
  void applyVisitor(IMultiArray2DataT<Real2>*) override {}
  void applyVisitor(IMultiArray2DataT<Real3>*) override {}
  void applyVisitor(IMultiArray2DataT<Real2x2>*) override {}
  void applyVisitor(IMultiArray2DataT<Real3x3>*) override {}

 private:

  IDataTypeDataDispatcherT<Byte>* m_byte;
  IDataTypeDataDispatcherT<Real>* m_real;
  IDataTypeDataDispatcherT<Int16>* m_int16;
  IDataTypeDataDispatcherT<Int32>* m_int32;
  IDataTypeDataDispatcherT<Int64>* m_int64;
  IDataTypeDataDispatcherT<Real2>* m_real2;
  IDataTypeDataDispatcherT<Real3>* m_real3;
  IDataTypeDataDispatcherT<Real2x2>* m_real2x2;
  IDataTypeDataDispatcherT<Real3x3>* m_real3x3;
  IDataTypeDataDispatcherT<String>* m_string;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief IDataVisitor dispatching operations according to the data type.
 *
 * The dispatcher must inherit from IDataTypeDataDispatcherT and the interface
 * \a IDispatcherType passed as an argument to the template class.
 *
 * The \a IDispatcherType interface must define the type
 * HasStringDispatch as TrueType or FalseType depending on whether it
 * supports String type dispatch.
 *
 * It is possible to construct an instance directly via
 * the constructor or via the static function create().
 */
template<typename IDispatcherType>
class DataTypeDispatchingDataVisitor
: public AbstractDataTypeDispatchingDataVisitor
{
 public:
  typedef DataTypeDispatchingDataVisitor<IDispatcherType> ThatClass;
  typedef IDispatcherType* IDispatcherTypePtr;
 public:

  /*!
   * \brief Constructs an instance.
   * The a_* and i_* parameters must specify the same instance,
   * one via the IDataTypeDataDispatcherT interface and the other
   * via IDispatcherType.
   *
   * The objects passed as parameters become the property of this
   * instance, which is responsible for destroying them via the operator
   * delete in the destructor
   */
  DataTypeDispatchingDataVisitor(IDataTypeDataDispatcherT<Byte>* a_byte,
                                 IDataTypeDataDispatcherT<Real>* a_real,
                                 IDataTypeDataDispatcherT<Int16>* a_int16,
                                 IDataTypeDataDispatcherT<Int32>* a_int32,
                                 IDataTypeDataDispatcherT<Int64>* a_int64,
                                 IDataTypeDataDispatcherT<Real2>* a_real2,
                                 IDataTypeDataDispatcherT<Real3>* a_real3,
                                 IDataTypeDataDispatcherT<Real2x2>* a_real2x2,
                                 IDataTypeDataDispatcherT<Real3x3>* a_real3x3,
                                 IDataTypeDataDispatcherT<String>* a_string)
  : AbstractDataTypeDispatchingDataVisitor(a_byte,a_real,a_int16,a_int32,a_int64,a_real2,a_real3,
                                           a_real2x2,a_real3x3,a_string){}

 public:

  /*!
   * \brief Creates an instance of the class.
   *
   * Creates an instance using the concrete type \a TrueDispatcherType and taking
   * the type \a BuildArgType as an argument during construction.
   *
   * This operation performs the following for each type \a DataType:
   * - new TrueDispatcherType<DataType>(arg);
   */
  template<template<typename DataType> class TrueDispatcherType,typename BuildArgType> static ThatClass*
  create(BuildArgType arg)
  {
    typedef typename IDispatcherType::HasStringDispatch HasStringDispatch;
    return _create<TrueDispatcherType,BuildArgType>(arg,HasStringDispatch());
  }

 private:
  template<template<typename DataType> class TrueDispatcherType,typename BuildArgType> static ThatClass*
  _create(BuildArgType arg,TrueType)
  {
    TrueDispatcherType<String>* a_string = new TrueDispatcherType<String>(arg);
    return _create2(arg,a_string,a_string);
  }
  template<template<typename DataType> class TrueDispatcherType,typename BuildArgType> static ThatClass*
  _create(BuildArgType arg,FalseType)
  {
    return _create2<TrueDispatcherType,BuildArgType>(arg,0,0);
  }
  template<template<typename DataType> class TrueDispatcherType,typename BuildArgType> static ThatClass*
  _create2(BuildArgType arg,IDataTypeDataDispatcherT<String>* a_string,IDispatcherType* i_string)
  {
    TrueDispatcherType<Byte>* a_byte = new TrueDispatcherType<Byte>(arg);
    TrueDispatcherType<Real>* a_real = new TrueDispatcherType<Real>(arg);
    TrueDispatcherType<Int16>* a_int16 = new TrueDispatcherType<Int16>(arg);
    TrueDispatcherType<Int32>* a_int32 = new TrueDispatcherType<Int32>(arg);
    TrueDispatcherType<Int64>* a_int64 = new TrueDispatcherType<Int64>(arg);
    TrueDispatcherType<Real2>* a_real2 = new TrueDispatcherType<Real2>(arg);
    TrueDispatcherType<Real3>* a_real3 = new TrueDispatcherType<Real3>(arg);
    TrueDispatcherType<Real2x2>* a_real2x2 = new TrueDispatcherType<Real2x2>(arg);
    TrueDispatcherType<Real3x3>* a_real3x3 = new TrueDispatcherType<Real3x3>(arg);
    ThatClass* p = new ThatClass(a_byte,a_real,a_int16,a_int32,a_int64,a_real2,a_real3,
                                 a_real2x2,a_real3x3,a_string);
    p->setDispatchers(a_byte,a_real,a_int16,a_int32,a_int64,a_real2,a_real3,
                      a_real2x2,a_real3x3,i_string);
    return p;
  }

 public:

  //! List of dispatchers
  ConstArrayView<IDispatcherType*> dispatchers() const
  {
    return ConstArrayView<IDispatcherType*>(m_nb_dispatcher,m_dispatchers);
  }

 public:

  /*!
   *\brief Positions the list of dispatchers.
   * \warning should only be done during initialization. The method
   * create() calls this method automatically.
   */
  void setDispatchers(IDispatcherType* i_byte,
                      IDispatcherType* i_real,
                      IDispatcherType* i_int16,
                      IDispatcherType* i_int32,
                      IDispatcherType* i_int64,
                      IDispatcherType* i_real2,
                      IDispatcherType* i_real3,
                      IDispatcherType* i_real2x2,
                      IDispatcherType* i_real3x3,
                      IDispatcherType* i_string
                      )
  {
    m_nb_dispatcher = 10;

    m_dispatchers[0] = i_byte;
    m_dispatchers[1] = i_real;
    m_dispatchers[2] = i_int16;
    m_dispatchers[3] = i_int32;
    m_dispatchers[4] = i_int64;
    m_dispatchers[5] = i_real2;
    m_dispatchers[6] = i_real3;
    m_dispatchers[7] = i_real2x2;
    m_dispatchers[8] = i_real3x3;
    m_dispatchers[9] = i_string;
    if (!i_string)
      --m_nb_dispatcher;
  }

 protected:
  Integer m_nb_dispatcher = 0;
  IDispatcherTypePtr m_dispatchers[10] = { };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
