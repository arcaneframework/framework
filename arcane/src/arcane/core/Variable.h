// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Variable.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Classe gérant la partie privée d'une variable.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLE_H
#define ARCANE_CORE_VARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IData.h"

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
class VariableResizeArgs;

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
  IVariableMng* variableMng() const override;
  String fullName() const final;
  String name() const final;
  String itemFamilyName() const final;
  String meshName() const final;
  String itemGroupName() const final;
  int property() const override;
  void notifyReferencePropertyChanged() override;

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
  Ref<VariableMetaData> createMetaDataRef() const override;
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

  void setAllocationInfo(const DataAllocationInfo& v) override;
  DataAllocationInfo allocationInfo() const override;

 public:
  
  IObservable* writeObservable() override;
  IObservable* readObservable() override;
  IObservable* onSizeChangedObservable() override;

 public:

  void addTag(const String& tagname,const String& tagvalue) override;
  void removeTag(const String& tagname) override;
  bool hasTag(const String& tagname) override;
  String tagValue(const String& tagname) override;

 public:

  void update() override;
  void setUpToDate() override;
  Int64 modifiedTime() override;
  void addDepend(IVariable* var,eDependType dt) override;
  void addDepend(IVariable* var,eDependType dt,const TraceInfo& tinfo) override;
  void removeDepend(IVariable* var) override;
  void setComputeFunction(IVariableComputeFunction* v) override;
  IVariableComputeFunction* computeFunction() override;
  void dependInfos(Array<VariableDependInfo>& infos) override;

  void update(Real wanted_time) override;

  void changeGroupIds(Int32ConstArrayView old_to_new_ids) override;

  IVariableInternal* _internalApi() override;

 public:
  
  IMemoryAccessTrace* memoryAccessTrace() const override { return nullptr; }

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
  /*!
   * \brief Indique si les données de la variable sont valides.
   *
   * Les données sont valides à la fin d'un appel à setUsed().
   */
  bool _hasValidData() const;

 protected:

  virtual void _internalResize(const VariableResizeArgs& resize_args) =0;
  virtual Integer _checkIfSameOnAllReplica(IParallelMng* replica_pm,int max_print) =0;
  void _checkSwapIsValid(Variable* rhs);
  // Temporaire pour test libération mémoire
  bool _wantShrink() const;

  // Accès via VariablePrivate pour l'API interne
  friend class VariablePrivate;
  void _resize(const VariableResizeArgs& resize_args);

  //! Comparaison de valeurs entre variables
  virtual VariableComparerResults _compareVariable(const VariableComparerArgs& compare_args) =0;

 private:

  VariablePrivate* m_p; //!< Implémentation

 private:
  
  void _checkSetItemFamily();
  void _checkSetItemGroup();
  void _checkSetProperty(VariableRef*);
  bool _hasReference() const;
  void _removeMeshReference();
  String _computeComparisonHashCollective(IData* sorted_data);
  VariableMetaData* _createMetaData() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

#include "arcane/core/VariableScalar.h"
#include "arcane/core/VariableArray.h"
