// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionExtended.h                                        (C) 2000-2023 */
/*                                                                           */
/* Option du jeu de données de type 'Extended'.                              */
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
 * \brief Option du jeu de données de type étendu.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionExtended
: public CaseOptionSimple
{
 public:

  CaseOptionExtended(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionSimple(cob), m_type_name(type_name) {}

 public:

  void print(const String& lang,std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real /*current_time*/,Integer /*current_iteration*/) override {}
  void visit(ICaseDocumentVisitor* visitor) const override;

  /*!
   * \brief Positionne la valeur par défaut de l'option.
   *
   * Si l'option n'est pas pas présente dans le jeu de données, alors sa valeur sera
   * celle spécifiée par l'argument \a def_value, sinon l'appel de cette méthode est sans effet.
   */
  void setDefaultValue(const String& def_value);

 protected:

  virtual bool _tryToConvert(const String& s) =0;
  
  void _search(bool is_phase1) override;
  bool _allowPhysicalUnit() override { return false; }

  String _typeName() const { return m_type_name; }

 private:

  String m_type_name; //!< Nom du type de l'option
  String m_value; //!< Valeur de l'option sous forme de chaîne unicode
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de donnée de type étendu.
 *
 * \ingroup CaseOption
 * Cette classe se sert d'une fonction externe dont le prototype est:
 
 \code
 extern "C++" bool
 _caseOptionConvert(const CaseOption&,const String&,T& obj);
 \endcode
 
 pour retrouver à partir d'une chaine de caractère un objet du type \a T.
 Cette fonction retourne \a true si un tel objet n'est pas trouvé.
 Si l'objet est trouvé, il est stocké dans \a obj.
 */
#ifndef SWIG
template<class T>
class CaseOptionExtendedT
: public CaseOptionExtended
{
 public:

  CaseOptionExtendedT(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionExtended(cob,type_name) {}

 public:

  //! Valeur de l'option
  operator const T&() const { return value(); }

  //! Valeur de l'option
  const T& value() const
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return m_value;
  }

  //! Valeur de l'option
  const T& operator()() const { return value(); }

  //! Retourne la valeur de l'option si isPresent()==true ou sinon \a arg_value
  const T& valueIfPresentOrArgument(const T& arg_value)
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return isPresent() ? m_value : arg_value;
  }

 protected:
	
  virtual bool _tryToConvert(const String& s)
  {
    // La fonction _caseOptionConvert() doit être déclarée avant
    // l'instantiation de cette template. Normalement le générateur automatique
    // de config effectue cette opération.
    return _caseOptionConvert(*this,s,m_value);
  }

 private:

  T m_value; //!< Valeur de l'option
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de donnée de type liste de types étendus.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionMultiExtended
: public CaseOptionBase
{
 public:

  CaseOptionMultiExtended(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionBase(cob), m_type_name(type_name) {}
  ~CaseOptionMultiExtended() {}

 public:

  void print(const String& lang,std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real /*current_time*/,Integer /*current_iteration*/) override {}
  void visit(ICaseDocumentVisitor* visitor) const override;

 protected:
  
  virtual bool _tryToConvert(const String& s,Integer pos) =0;
  virtual void _allocate(Integer size) =0;
  virtual bool _allowPhysicalUnit() { return false; }
  virtual Integer _nbElem() const =0;
  String _typeName() const { return m_type_name; } 
  void _search(bool is_phase1) override;

 private:

  String m_type_name; //!< Nom du type de l'option
  UniqueArray<String> m_values; //!< Valeurs sous forme de chaînes unicodes.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de donnée de type liste de types étendus.
 * \ingroup CaseOption
 * \warning Toutes les méthodes de cette classe doivent être visible dans la
 * déclaration (pour éviter des problèmes d'instanciation de templates).
 * \sa CaseOptionExtendedT
 */
#ifndef SWIG
template<class T>
class CaseOptionMultiExtendedT
: public CaseOptionMultiExtended
, public ArrayView<T>
{
 public:

  typedef T Type; //!< Type de l'option.

 public:

  CaseOptionMultiExtendedT(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionMultiExtended(cob,type_name) {}
  virtual ~CaseOptionMultiExtendedT() {} // delete[] _ptr(); }

 public:

 protected:

  bool _tryToConvert(const String& s,Integer pos) override
  {
    // La fonction _caseOptionConvert() doit être déclarée avant
    // l'instantiation de cette template. Normalement le générateur automatique
    // d'options (axl2cc) effectue cette opération.
    T& value = this->operator[](pos);
    return _caseOptionConvert(*this,s,value);
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
