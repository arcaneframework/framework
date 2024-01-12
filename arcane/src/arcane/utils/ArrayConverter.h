// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayConverter.h                                            (C) 2000-2023 */
/*                                                                           */
/* Conversion d'un tableau d'un type vers un autre type.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYCONVERTER_H
#define ARCANE_UTILS_ARRAYCONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename TypeA,typename TypeB>
class DefaultConverter
{
 public:
  void convertFromAToB(ConstArrayView<TypeA> input,ArrayView<TypeB> output)
  {
    for( Integer i=0, is=input.size(); i<is; ++i )
      output[i] = (TypeB)input[i];
  }
  void convertFromBToA(ConstArrayView<TypeB> input,ArrayView<TypeA> output)
  {
    for( Integer i=0, is=input.size(); i<is; ++i )
      output[i] = (TypeA)input[i];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conversion d'un tableau d'un type vers un autre type.
 */
template<typename InputType,typename OutputType,
  typename Converter = DefaultConverter<InputType,OutputType> >
class ArrayConverter
{
 public:
  typedef UniqueArray<OutputType> OutputArrayType;
  typedef typename OutputArrayType::iterator iterator;
  typedef typename OutputArrayType::const_iterator const_iterator;
  typedef typename OutputArrayType::pointer pointer;
  typedef typename OutputArrayType::const_pointer const_pointer;
 public:

  ArrayConverter() {}
  ArrayConverter(Converter & conv)
  : m_converter(conv) 
  {
  }

  ArrayConverter(Integer nb,InputType* values)
  {
    m_input_array = ArrayView<InputType>(nb,values);
    _init();
  }
  ArrayConverter(ArrayView<InputType> values)
  {
    m_input_array = values;
    _init();
  }
  ArrayConverter(ArrayView<InputType> values, Converter& conv)
  :m_converter(conv)
  {
    m_input_array = values;
    _init();
  }
  ~ArrayConverter() noexcept(false)

  {
    m_converter.convertFromBToA(m_output_array,m_input_array);
  }
  void operator=(ArrayView<InputType> values)
  {
    m_input_array = values;
    _init();
  }
  void notifyOutputChanged()
  {
    m_converter.convertFromBToA(m_output_array,m_input_array);
  }
  void notifyInputChanged()
  {
    m_converter.convertFromAToB(m_input_array,m_output_array);
  }
 public:

  /*!
   * \deprecated Utiliser data() à la place.
   */
  ARCANE_DEPRECATED_280 iterator begin() { return m_output_array.begin(); }
  /*!
   * \deprecated Utiliser data() à la place.
   */
  ARCANE_DEPRECATED_280 const_iterator begin() const { return m_output_array.begin(); }
  OutputArrayType& array() { return m_output_array; }
  OutputArrayType& array() const { return m_output_array; }
  pointer data() { return m_output_array.data(); }
  const_pointer data() const { return m_output_array.data(); }

 private:

  void _init() 
  {
    m_output_array.resize(m_input_array.size());
    m_converter.convertFromAToB(m_input_array,m_output_array);
  }

  ArrayView<InputType> m_input_array;
  OutputArrayType m_output_array;
  Converter m_converter;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conversion d'un tableau d'un type vers un autre type.
 *
 * Spécialisation pour le cas ou le type d'entré et de sortie est
 * le même.
 */
template<typename InputType>
class ArrayConverter<InputType,InputType,DefaultConverter<InputType,InputType> >
{
 public:
  typedef ArrayView<InputType> OutputArrayType;
  typedef typename OutputArrayType::iterator iterator;
  typedef typename OutputArrayType::const_iterator const_iterator;
  typedef typename OutputArrayType::pointer pointer;
  typedef typename OutputArrayType::const_pointer const_pointer;
 public:

  ArrayConverter()
  {
  }
  ArrayConverter(Integer nb,InputType* values)
  {
    m_input_array = ArrayView<InputType>(nb,values);
  }
  ArrayConverter(ArrayView<InputType> values)
  {
    m_input_array = values;
  }
  ~ArrayConverter(){}
  void operator=(ArrayView<InputType> values)
  {
    m_input_array = values;
  }

 public:

  /*!
   * \deprecated Utiliser data() à la place.
   */
  ARCANE_DEPRECATED_280 iterator begin() { return m_input_array.begin(); }
  /*!
   * \deprecated Utiliser data() à la place.
   */
  ARCANE_DEPRECATED_280 const_iterator begin() const { return m_input_array.begin(); }
  OutputArrayType& array() { return m_input_array; }
  OutputArrayType& array() const { return m_input_array; }
  void notifyOutputChanged() {}
  void notifyInputChanged() {}
  pointer data() { return m_input_array.data(); }
  const_pointer data() const { return m_input_array.data(); }

 private:

  ArrayView<InputType> m_input_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conversion d'un tableau d'un type vers un autre type.
 */
template<typename InputType,typename OutputType,
  typename Converter = DefaultConverter<InputType,OutputType> >
class ConstArrayConverter
{
 public:
  typedef ConstArrayView<OutputType> OutputArrayType;
  typedef typename OutputArrayType::const_iterator const_iterator;
  typedef typename OutputArrayType::const_pointer const_pointer;
 public:

  ConstArrayConverter()
  {
  }

  ConstArrayConverter(Converter & conv)
  : m_converter(conv) 
  {
  }

  ConstArrayConverter(Integer nb,const InputType* values)
  {
    m_input_array = ConstArrayView<InputType>(nb,values);
    _init();
  }
  ConstArrayConverter(ConstArrayView<InputType> values)
  {
    m_input_array = values;
    _init();
  }
  ConstArrayConverter(ConstArrayView<InputType> values, Converter& conv)
  :m_converter(conv)
  {
    m_input_array = values;
    _init();
  }

  ~ConstArrayConverter() {}

 public:

  /*!
   * \deprecated Utiliser data() à la place.
   */
  ARCANE_DEPRECATED_280 const_iterator begin() const { return m_output_array.begin(); }
  OutputArrayType& array() const { return m_output_array; }
  const_pointer data() const { return m_output_array.data(); }

 private:

  void _init()
  {
    m_output_array.resize(m_input_array.size());
    m_converter.convertFromAToB(m_input_array,m_output_array);   
  }

  ConstArrayView<InputType> m_input_array;
  UniqueArray<OutputType> m_output_array;
  Converter m_converter;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conversion d'un tableau d'un type vers un autre type.
 *
 * Spécialisation pour le cas ou le type d'entrée et de sortie est
 * le même.
 */
template<typename InputType>
class ConstArrayConverter<InputType,InputType,DefaultConverter<InputType,InputType> >
{
 public:
  typedef ConstArrayView<InputType> OutputArrayType;
  typedef typename OutputArrayType::const_iterator const_iterator;
  typedef typename OutputArrayType::const_pointer const_pointer;
 public:

  ConstArrayConverter()
  {
  }
  ConstArrayConverter(Integer nb,const InputType* values)
  {
    m_input_array = ConstArrayView<InputType>(nb,values);
  }

  ConstArrayConverter(ConstArrayView<InputType> values)
  {
    m_input_array = values;
  }
  ~ConstArrayConverter()
  {
  }

 public:

  /*!
   * \deprecated Utiliser data() à la place.
   */
  ARCANE_DEPRECATED_280 const_iterator begin() const { return m_input_array.begin(); }
  OutputArrayType& array() const { return m_input_array; }
  const_pointer data() const { return m_input_array.data(); }

 private:

  ConstArrayView<InputType> m_input_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

