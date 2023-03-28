// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.h                                               (C) 2000-2022 */
/*                                                                           */
/* Options du jeu de données.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONS_H
#define ARCANE_CASEOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"

#include "arccore/base/ReferenceCounter.h"

#include "arcane/core/XmlNode.h"
#include "arcane/core/ICaseOptions.h"
#include "arcane/core/ICaseOptionList.h"
#include "arcane/core/CaseOptionBase.h"
#include "arcane/core/CaseOptionEnum.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInfo;
class IModule;
class ICaseOptions;
class ICaseFunction;
class ICaseDocument;
class ICaseMng;
class IScriptImpl;
class XmlNodeList;

class CaseOptionBuildInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une valeur d'une option complexe.
 *
 * Une option complexe est composé de plusieurs instances de cette classe.
 */
class ARCANE_CORE_EXPORT CaseOptionComplexValue
{
 public:

  CaseOptionComplexValue(ICaseOptionsMulti* opt,ICaseOptionList* clist,const XmlNode& parent_elem);
  virtual ~CaseOptionComplexValue();

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not access XML item from option")
  XmlNode element() const { return m_element; }

  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane. Do not use it")
  ICaseOptionList* configList() const { return m_config_list.get(); }

  //! Nom complet au format donné par la norme XPath.
  String xpathFullName() const { return m_element.xpathFullName(); }

 protected:

  // Les deux méthodes suivantes sont utilisés par le générateur 'axl2cc' et
  // ne doivent pas être modifiées.
  ICaseOptionList* _configList() { return m_config_list.get(); }
  XmlNode _element() { return m_element; }

 private:

  ReferenceCounter<ICaseOptionList> m_config_list;
  XmlNode m_element;
};

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

class CaseOptionsPrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une liste d'options du jeu de données.
 *
 * Les instances de cette classe doivent toutes être allouées par
 * l'opérateur new() et ne doivent pas être détruite, le gestionnaire
 * de cas (ICaseMng) s'en chargeant.
 */
class ARCANE_CORE_EXPORT CaseOptions
: public ICaseOptions
{
 public:
	
  //! Construit un jeu d'options.
  CaseOptions(ICaseMng* cm,const String& name);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*,const String& name);
  //! Construit un jeu d'options.
  CaseOptions(ICaseMng* cm,const String& name,const XmlNode& parent);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*,const String& name,const XmlNode& parent,bool is_optional=false,bool is_multi=false);
  //! Construit un jeu d'options.
 protected:
  CaseOptions(ICaseMng*,const String& name,ICaseOptionList* parent);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*,const String& name,ICaseOptionList* parent);
 public:
  //! Libère les ressources
  ~CaseOptions() override;

 private:
 
  CaseOptions(const CaseOptions& rhs) = delete;
  CaseOptions& operator=(const CaseOptions& rhs) = delete;

 public:

  //! Retourne le vrai nom (non traduit) de l'élément. 
  String rootTagTrueName() const override;

  //! Retourne le nom de l'élément dans le langage du jeu de données.
  String rootTagName() const override;

  //! Nom dans la langue \a lang de l'option. Retourne \a rootTagTrueName() si pas de traduction.
  String translatedName(const String& lang) const override;

  //! Retourne le vrai nom (non traduit) de l'élément. 
  virtual String trueName() const { return rootTagTrueName(); }

  //! Retourne le nom de l'élément dans le langage du jeu de données.
  virtual String name() const { return rootTagName(); }

  void read(bool is_phase1) override
  {
    auto p = (is_phase1) ? eCaseOptionReadPhase::Phase1 : eCaseOptionReadPhase::Phase2;
    read(p);
  }

  void read(eCaseOptionReadPhase phase) override;

  void addInvalidChildren(XmlNodeList&) override;

  void printChildren(const String& lang,int indent) override;

  //! Retourne le service associé ou 0 s'il n'y en a pas.
  IServiceInfo* caseServiceInfo() const override;

  //! Retourne le module associé ou 0 s'il n'y en a pas.
  IModule* caseModule() const override;

  //! Associe le service \a m à ce jeu de données.
  void setCaseServiceInfo(IServiceInfo* m) override;

  //! Associe le module \a m à ce jeu de données.
  void setCaseModule(IModule* m) override;

  //! Ajoute à la liste \a col tous les options filles.
  void deepGetChildren(Array<CaseOptionBase*>& col) override;

  ICaseOptionList* configList() override;

  const ICaseOptionList* configList() const override;

  //! Fonction indiquant l'état d'activation de l'option
  ICaseFunction* activateFunction() override;

  /*!
    \brief Vrai si l'option est présente dans le fichier,
    faux s'il s'agit de la valeur par défaut.
  */
  bool isPresent() const override;
  
  /*!
    \brief Ajoute une traduction pour le nom de l'option.
    Ajoute le nom \a name de l'option correspondant au langage \a lang.
    Si une traduction existe déjà pour ce langage, elle est remplacée par
    celle-ci.
  */
  void addAlternativeNodeName(const String& lang,const String& name) override;

  ICaseMng* caseMng() const override;
  ITraceMng* traceMng() const override;
  ISubDomain* subDomain() const override;
  IMesh* mesh() const override;
  MeshHandle meshHandle() const override;
  ICaseDocument* caseDocument() const override;

  void detach() override;

  void visit(ICaseDocumentVisitor* visitor) const override;

  String xpathFullName() const override;

 public:

  void addReference() override;
  void removeReference() override;

 protected:

  void _setTranslatedName();
  bool _setMeshHandleAndCheckDisabled(const String& mesh_name);

 protected:

  CaseOptionsPrivate* m_p; //!< Implémentation

 private:

  void _setMeshHandle(const MeshHandle& handle);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
