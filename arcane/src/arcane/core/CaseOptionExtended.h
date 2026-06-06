// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionExtended.h                                        (C) 2000-2023 */
/*                                                                           */
/* Option for the dataset of type 'Extended'.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONEXTENDED_H
#define ARCANE_CASEOPTIONEXTENDED_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/CaseOptionSimple.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Option for the extended type dataset.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionExtended
: public CaseOptionSimple
{
 public:

  CaseOptionExtended(const CaseOptionBuildInfo& cob, const String& type_name)
  : CaseOptionSimple(cob)
  , m_type_name(type_name)
  {}

 public:

  void print(const String& lang, std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real /*current_time*/, Integer /*current_iteration*/) override {}
  void visit(ICaseDocumentVisitor* visitor) const override;

  /*!
   * \brief Sets the default value of the option.
   *
   * If the option is not present in the dataset, its value will be
   * that specified by the argument \a def_value; otherwise, calling this method has no effect.
   */
  void setDefaultValue(const String& def_value);

 protected:

  virtual bool _tryToConvert(const String& s) = 0;

  void _search(bool is_phase1) override;
  bool _allowPhysicalUnit() override { return false; }

  String _typeName() const { return m_type_name; }

 private:

  String m_type_name; //!< Name of the option type
  String m_value; //!< Value of the option in unicode string format
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Option for the extended type dataset.
 *
 * \ingroup CaseOption
 * This class uses an external function with the prototype:
 
 \code
 extern "C++" bool
 _caseOptionConvert(const CaseOption&,const String&,T& obj);
 \endcode
 
 to retrieve an object of type \a T from a character string.
 This function returns \a true if such an object is not found.
 If the object is found, it is stored in \a obj.
 */
#ifndef SWIG
template <class T>
class CaseOptionExtendedT
: public CaseOptionExtended
{
 public:

  CaseOptionExtendedT(const CaseOptionBuildInfo& cob, const String& type_name)
  : CaseOptionExtended(cob, type_name)
  {}

 public:

  //! Option value
  operator const T&() const { return value(); }

  //! Option value
  const T& value() const
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return m_value;
  }

  //! Option value
  const T& operator()() const { return value(); }

  //! Returns the value of the option if isPresent()==true or otherwise \a arg_value
  const T& valueIfPresentOrArgument(const T& arg_value)
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return isPresent() ? m_value : arg_value;
  }

 protected:

  virtual bool _tryToConvert(const String& s)
  {
    // The _caseOptionConvert() function must be declared before
    // the instantiation of this template. Normally, the automatic config generator
    // performs this operation.
    return _caseOptionConvert(*this, s, m_value);
  }

 private:

  T m_value; //!< Value of the option
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Option for the extended list of types dataset.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionMultiExtended
: public CaseOptionBase
{
 public:

  CaseOptionMultiExtended(const CaseOptionBuildInfo& cob, const String& type_name)
  : CaseOptionBase(cob)
  , m_type_name(type_name)
  {}
  ~CaseOptionMultiExtended() {}

 public:

  void print(const String& lang, std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real /*current_time*/, Integer /*current_iteration*/) override {}
  void visit(ICaseDocumentVisitor* visitor) const override;

 protected:

  virtual bool _tryToConvert(const String& s, Integer pos) = 0;
  virtual void _allocate(Integer size) = 0;
  virtual bool _allowPhysicalUnit() { return false; }
  virtual Integer _nbElem() const = 0;
  String _typeName() const { return m_type_name; }
  void _search(bool is_phase1) override;

 private:

  String m_type_name; //!< Name of the option type
  UniqueArray<String> m_values; //!< Values in unicode string format.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Option for the extended list of types dataset.
 * \ingroup CaseOption
 * \warning All methods of this class must be visible in the declaration (to avoid template instantiation problems).
 * \sa CaseOptionExtendedT
 */
#ifndef SWIG
template <class T>
class CaseOptionMultiExtendedT
: public CaseOptionMultiExtended
, public ArrayView<T>
{
 public:

  typedef T Type; //!< Type of the option.

 public:

  CaseOptionMultiExtendedT(const CaseOptionBuildInfo& cob, const String& type_name)
  : CaseOptionMultiExtended(cob, type_name)
  {}
  virtual ~CaseOptionMultiExtendedT() {} // delete[] _ptr(); }

 public:
 protected:

  bool _tryToConvert(const String& s, Integer pos) override
  {
    // The _caseOptionConvert() function must be declared before
    // the instantiation of this template. Normally, the automatic config generator
    // of options (axl2cc) performs this operation.
    T& value = this->operator[](pos);
    return _caseOptionConvert(*this, s, value);
  }
  void _allocate(Integer size) override
  {
    m_values.resize(size);
    ArrayView<T>* view = this;
    *view = m_values.view();
  }
  //virtual const void* _elemPtr(Integer i) const { return this->begin()+i; }
  virtual Integer _nbElem() const override { return m_values.size(); }

 private:

  UniqueArray<T> m_values;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
