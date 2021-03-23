// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariable.h                                     (C) 2000-2018 */
/*                                                                           */
/* Interface d'un variable sur un matériau du maillage.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALVARIABLE_H
#define ARCANE_MATERIALS_IMESHMATERIALVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IData;
class VariableDependInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialVariableMng;
class IMeshEnvironment;
class IMeshMaterial;
class MeshMaterialVariableRef;
class MeshMaterialVariableDependInfo;
class IMeshMaterialVariableComputeFunction;
class AllEnvCellVectorView;
class ComponentItemListBuilder;
class MeshMaterialVariableSynchronizerList;
class IMeshComponent;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une variable matériau d'un maillage.
 */
class IMeshMaterialVariable
{
 public:

  virtual ~IMeshMaterialVariable(){}

 public:

  //! Nom de la variable.
  virtual const String& name() const =0;

  //! Variable globale sur le maillage associée.
  virtual IVariable* globalVariable() const =0;

  /*!
   * \internal
   * \brief Construit les infos de la variable.
   * A usage interne à Arcane.
   */
  virtual void buildFromManager(bool is_continue) =0;

  //! \internal
  virtual IData* _createSaveData(Integer nb_value) =0;

  //! \internal
  virtual Ref<IData> _internalCreateSaveDataRef(Integer nb_value) =0;

  //! \internal
  virtual void _saveData(IMeshComponent* component,IData* data) =0;

  //! \internal
  virtual void _restoreData(IMeshComponent* component,IData* data,Integer data_index,Int32ConstArrayView ids,bool allow_null_id) =0;

  //! \internal
  virtual void _copyGlobalToPartial(Int32 var_index,Int32ConstArrayView local_ids,Int32ConstArrayView indexes_in_multiple) =0;

  //! \internal
  virtual void _copyPartialToGlobal(Int32 var_index,Int32ConstArrayView local_ids,Int32ConstArrayView indexes_in_multiple) =0;

  //! \internal
  virtual void _initializeNewItems(const ComponentItemListBuilder& list_builder) =0;

  /*!
   * \brief Synchronise les références.
   *
   * Synchronise les valeurs des références (VariableRef) à cette variable
   * avec la valeur actuelle de la variable. Cette méthode est appelé
   * automatiquement lorsque le nombre d'éléments d'une variable tableau change.
   */
  virtual void syncReferences() =0;

  /*!
   * \brief Ajoute une référence à cette variable
   *
   * \pre \a var_ref ne doit pas déjà référencer une variable.
   */
  virtual void addVariableRef(MeshMaterialVariableRef* var_ref) =0;

  /*!
   * \brief Supprime une référence à cette variable
   *
   * \pre \a var_ref doit référencer cette variable (un appel à addVariableRef()
   * doit avoir été effectué sur cette variable).
   */
  virtual void removeVariableRef(MeshMaterialVariableRef* var_ref) =0;

  //! \internal
  virtual MeshMaterialVariableRef* firstReference() const =0;

  /*!
   * \internal
   * \brief Variable contenant les valeurs spécifiques du matériau \a mat.
   */
  virtual IVariable* materialVariable(IMeshMaterial* mat) =0;

  /*!
   * \brief Indique si on souhaite conserver la valeur de la variable
   * apres un changement de la liste des matériaux.
   */
  virtual void setKeepOnChange(bool v) =0;

  /*!
   * \brief Indique si on souhaite conserver la valeur de la variable
   * apres un changement de la liste des matériaux.
   */
  virtual bool keepOnChange() const =0;

  /*!
   * \brief Synchronise la variable.
   *
   * La synchronisation se fait sur tous les matériaux de la maille.
   * Il est indispensable que toutes mailles fantômes aient déjà le bon
   * nombre de matériaux.
   */
  virtual void synchronize() =0;

  virtual void synchronize(MeshMaterialVariableSynchronizerList& sync_list) =0;

  /*!
   * \brief Affiche les valeurs de la variable sur le flot \a ostr.
   */
  virtual void dumpValues(ostream& ostr) =0;

  /*!
   * \brief Affiche les valeurs de la variable pour la vue \a view sur le flot \a ostr.
   */
  virtual void dumpValues(ostream& ostr,AllEnvCellVectorView view) =0;

  /*!
   * \brief Remplit les valeurs partielles avec la valeur de la maille globale associée.
   */
  virtual void fillPartialValuesWithGlobalValues() =0;

  /*!
   * \brief Remplit les valeurs partielles avec la valeur de la maille du dessus.
   * Si \a level vaut LEVEL_MATERIAL, copie les valeurs matériaux avec celle du milieu.
   * Si \a level vaut LEVEL_ENVIRONNEMENT, copie les valeurs des milieux avec
   * celui de la maille globale.
   * Si \a level vaut LEVEL_ALLENVIRONMENT, remplit toutes les valeurs partielles
   * avec celle de la maille globale (cela rend cette méthode équivalente à
   * fillGlobalValuesWithGlobalValues().
   */
  virtual void fillPartialValuesWithSuperValues(Int32 level) =0;

  //! Sérialise la variable pour les entités de numéro local \a ids
  virtual void serialize(ISerializer* sbuffer,Int32ConstArrayView ids) =0;

  //! Espace de définition de la variable (matériau+milieu ou milieu uniquement)
  virtual MatVarSpace space() const =0;

 public:

  //! @name Gestion des dépendances
  //@{
  /*! \brief Recalcule la variable pour le matériau \a mat si nécessaire
   *
   * Par le mécanisme de dépendances, cette opération est appelée récursivement
   * sur toutes les variables dont dépend l'instance. La fonction de recalcul
   * computeFunction() est ensuite appelée s'il s'avère qu'une des variables
   * dont elle dépend a été modifiée plus récemment.
   *
   * \pre computeFunction() != 0
   */
  virtual void update(IMeshMaterial* mat) =0;

  /*! \brief Indique que la variable vient d'être mise à jour.
   *
   * Pour une gestion correcte des dépendances, il faut que cette propriété
   * soit appelée toutes les fois où la mise à jour d'une variable a été
   * effectuée.
   */
  virtual void setUpToDate(IMeshMaterial* mat) =0;

  //! Temps auquel la variable a été mise à jour
  virtual Int64 modifiedTime(IMeshMaterial* mat) =0;

  //! Ajoute \a var à la liste des dépendances
  virtual void addDepend(IMeshMaterialVariable* var) =0;

  //! Ajoute \a var à la liste des dépendances avec les infos de trace \a tinfo
  virtual void addDepend(IMeshMaterialVariable* var,const TraceInfo& tinfo) =0;

  //! Ajoute \a var à la liste des dépendances
  virtual void addDepend(IVariable* var) =0;

  //! Ajoute \a var à la liste des dépendances avec les infos de trace \a tinfo
  virtual void addDepend(IVariable* var,const TraceInfo& tinfo) =0;

  /*! \brief Supprime \a var de la liste des dépendances
   */
  virtual void removeDepend(IMeshMaterialVariable* var) =0;

  /*! \brief Supprime \a var de la liste des dépendances
   */
  virtual void removeDepend(IVariable* var) =0;

  /*! \brief Positionne la fonction de recalcule de la variable.
   *
   * Si une fonction de recalcule existait déjà, elle est détruite
   * et remplacée par celle-ci.
   */
  virtual void setComputeFunction(IMeshMaterialVariableComputeFunction* v) =0;

  //! Fonction utilisée pour mettre à jour la variable
  virtual IMeshMaterialVariableComputeFunction* computeFunction() =0;

  /*!
   * \brief Infos de dépendances.
   *
   * Remplit le tableau \a infos avec les infos de dépendance sur les variables
   * globales et le tableau \a mat_infos avec celles sur les variables matériaux.
   */
  virtual void dependInfos(Array<VariableDependInfo>& infos,
                           Array<MeshMaterialVariableDependInfo>& mat_infos) =0;
  //@}

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

