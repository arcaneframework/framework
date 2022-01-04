// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Variable.h                                                  (C) 2000-2020 */
/*                                                                           */
/* Classe gérant la partie privée d'une variable.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLE_H
#define ARCANE_VARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/IVariable.h"
#include "arcane/IData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableValue;
class VariableInfo;
class VariableBuildInfo;
template<typename T> class IDataTracerT;
class VariablePrivate;
class MemoryAccessInfo;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Instance d'une variable.
 *
 Cette classe gère les données d'une variable. Cette instance ne doit en
 principe pas être utilisée par les développeurs de code.
 \todo expliquer mieux
	
 Une variable est caractérisée par:
 <ul>
 <li>son <b>nom</b>,</li>
 <li>son <b>type</b>: réel, entier, tenseur, ...</li>,
 <li>son <b>genre</b>: scalaire, tableau, grandeur au noeud, grandeur
 au centre des mailles ...</li>
 </ul>

 Une variable qui repose sur un type d'entité du maillage est appelée une
 variable du maillage.

 La variable est généralement utilisée par un module (IModule) via une référence
 (VariableRef).

 Les variables sont <b>persistantes</b> et leur lecture/écriture se fait
 par les méthodes read() et write().

 \warning Cette classe est gérée entièrement par Arcane et les modules
 qui l'utilisent ne doivent en principe ne l'utiliser que pour récupérer des
 informations. Les opérations qui modifient cette instance (comme setItemGroup())
 ne doivent être utilisé que si le développeur possède une bonne connaissance
 de leur fonctionnement.

 * Cette classe ne doit pas être copiée. 
 */
class ARCANE_CORE_EXPORT Variable
: public TraceAccessor
, public IVariable
{
 protected:

  //! Créé une variable lié à la référence \a v.
  Variable(const VariableBuildInfo& v,const VariableInfo& vi);
 
 public:
  
  //! Libère les ressources
  ~Variable() override;

 private:
  
  //! Constructeur de recopie (ne pas utiliser)
  Variable(const Variable& from) = delete;
  //! Opérateur de recopie (ne pas utiliser)
  Variable& operator=(const Variable& from) = delete;

 public:

  ISubDomain* subDomain() override;
  const String& fullName() const override;
  const String& name() const override;
  const String& itemFamilyName() const override;
  const String& meshName() const override;
  const String& itemGroupName() const override;
  int property() const override;
  void notifyReferencePropertyChanged() override;

 public:

  Expression expression() override;

 public:
	
  //! Positionne l'état d'utilisation de la variable
  void setUsed(bool v) override;

  //! Etat d'utilisation de la variable
  bool isUsed() const override;

  bool isPartial() const override;

 public:

  void setTraceInfo(Integer,eTraceType) override {}

 public:
  
  void read(IDataReader* d) override;
  void write(IDataWriter* d) override;
  void notifyEndRead() override;
  void notifyBeginWrite() override;

 public:

  void addVariableRef(VariableRef* ref) override;
  void removeVariableRef(VariableRef* ref) override;
  VariableRef* firstReference() const override;
  Integer nbReference() const override;

  VariableMetaData* createMetaData() const override;
  void syncReferences() override;

  IMesh* mesh() const final;
  MeshHandle meshHandle() const final;
  ItemGroup itemGroup() const final;
  IItemFamily* itemFamily() const final;

  eItemKind itemKind() const override;
  Integer dimension() const override;
  Integer multiTag() const override;
  Integer checkIfSync(Integer max_print) override;
  Integer checkIfSameOnAllReplica(Integer max_print) override;

  eDataType dataType() const override;
  bool initialize(const ItemGroup& /*group*/,const String& /*value*/) override { return true; }

  IDataFactoryMng* dataFactoryMng() const final;
  void serialize(ISerializer* sbuffer,IDataOperation* operation) override;
  void serialize(ISerializer* sbuffer,Int32ConstArrayView ids,IDataOperation* operation) override;

  void resize(Integer n) override;
  void resizeFromGroup() override;

 public:
  
  virtual IObservable* writeObservable() override;
  virtual IObservable* readObservable() override;
  virtual IObservable* onSizeChangedObservable() override;

 public:

  virtual void addTag(const String& tagname,const String& tagvalue) override;
  virtual void removeTag(const String& tagname) override;
  virtual bool hasTag(const String& tagname) override;
  virtual String tagValue(const String& tagname) override;

 public:

  virtual void update() override;
  virtual void setUpToDate() override;
  virtual Int64 modifiedTime() override;
  virtual void addDepend(IVariable* var,eDependType dt) override;
  virtual void addDepend(IVariable* var,eDependType dt,const TraceInfo& tinfo) override;
  virtual void removeDepend(IVariable* var) override;
  virtual void setComputeFunction(IVariableComputeFunction* v) override;
  virtual IVariableComputeFunction* computeFunction() override;
  virtual void dependInfos(Array<VariableDependInfo>& infos) override;

  virtual void update(Real wanted_time) override;

  virtual void changeGroupIds(Int32ConstArrayView old_to_new_ids) override;

 public:
  
  virtual IMemoryAccessTrace* memoryAccessTrace() const override;

 protected:

  void _setProperty(int property);

  /*!
   * \brief Positionne la donnée.
   *
   * Si data est nul, une erreur fatale est envoyée
   */
  void _setData(const Ref<IData>& data);

  //! Indique si les données de la variable sont valides
  void _setValidData(bool valid_data);

  bool _wantAccessInfo();

 protected:

  virtual void _internalResize(Integer new_size,Integer nb_additional_element) =0;
  virtual Integer _checkIfSameOnAllReplica(IParallelMng* replica_pm,int max_print) =0;
  void _checkSwapIsValid(Variable* rhs);
  // Temporaire pour test libération mémoire
  bool _wantShrink() const;

 private:

  VariablePrivate* m_p; //!< Implémentation

 private:
  
  void _checkSetItemFamily();
  void _checkSetItemGroup();
  void _checkSetProperty(VariableRef*);
  bool _hasReference() const;
  void _removeMeshReference();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

#include "arcane/VariableScalar.h"
#include "arcane/VariableArray.h"
