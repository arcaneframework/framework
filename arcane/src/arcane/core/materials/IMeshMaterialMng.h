// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialMng.h                                          (C) 2000-2024 */
/*                                                                           */
/* Interface du gestionnaire des matériaux d'un maillage.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALMNG_H
#define ARCANE_MATERIALS_IMESHMATERIALMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IFunctorWithArgument.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Interface du gestionnaire des matériaux et des milieux d'un maillage.
 *
 * Cette interface gère les différents composants (IMeshComponent)
 * multi-matériaux d'un maillage  ainsi que leurs variables associées.
 * Ces composants peuvent être soit des matériaux (IMeshMaterial),
 * soit des milieux (IMeshEnvironment). Il est possible de récupérer la liste
 * des matériaux via materials() et la liste des milieux par environments().
 * Il est aussi possible de récupérer l'une de ces deux listes sous forme
 * 
 * L'implémentation actuelle ne gère que les matériaux et les milieux aux mailles.
 *
 * Une fois cette instance créé, via getReference(), la première chose à
 * faire est d'enregistrer
 * la liste des matériaux via registerMaterialInfo(). Il est ensuite
 * possible de créer chaque milieu en indiquant la liste des matériaux
 * qui le compose via createEnvironment(). Une fois ceci terminé, il
 * faut appeler endCreate() pour terminer l'initialisation. La liste des matériaux et des milieux
 * ne peut être modifiée que lors de l'initialisation. Elle ne doit plus évoluer
 * par la suite.
 *
 * Toute modification de la liste des mailles d'un milieu ou d'un matériau
 * doit se faire via une instance de MeshMaterialModifier.
 */
class ARCANE_CORE_EXPORT IMeshMaterialMng
{
  friend class MeshMaterialMngFactory;

 public:

  virtual ~IMeshMaterialMng() = default;

 public:

  /*!
   * \brief Récupère ou créé la référence associée à \a mesh.
   *
   * Si aucun gestionnaire de matériau n'est associé à \a mesh, il
   * sera créé lors de l'appel à cette méthode si \a create vaut \a true.
   * Si \a create vaut \a false est qu'aucune gestionnaire n'est associé
   * au maillage, un pointeur nul est retourné.
   * L'instance retournée reste valide tant que le maillage \a mesh existe.
   */
  static IMeshMaterialMng* getReference(const MeshHandleOrMesh& mesh_handle,bool create=true);

  /*!
   * \brief Récupère ou créé la référence associée à \a mesh.
   *
   * Si aucun gestionnaire de matériau n'est associé à \a mesh, il
   * sera créé lors de l'appel à cette méthode si \a create vaut \a true.
   * Si \a create vaut \a false est qu'aucune gestionnaire n'est associé
   * au maillage, un pointeur nul est retourné.
   * L'instance retournée reste valide tant que le maillage \a mesh existe.
   */
  static Ref<IMeshMaterialMng> getTrueReference(const MeshHandle& mesh_handle,bool create=true);

 public:

  //! Maillage associé.
  virtual IMesh* mesh() =0;

  //! Gestionnaire de traces
  virtual ITraceMng* traceMng() =0;

 public:

  /*!
   * \brief Enregistre les infos du matériau de nom \a name.
   *
   * Cette opération ne fait que enregistrer les informations d'un matériau.
   * Ces informations sont ensuite utilisés lors de la création du milieu
   * via createEnvironment().
   */
  virtual MeshMaterialInfo* registerMaterialInfo(const String& name) =0;

  /*!
   * \brief Créé un milieu avec les infos \a infos
   *
   * La création d'un milieu ne peut avoir lieu que lors de l'initialisation.
   * Les matériaux constituant le milieu doivent avoir auparavant été enregistrés via
   * \a registerMaterialInfo(). Un matériau peut appartenir à plusieurs milieux.
   */
  virtual IMeshEnvironment* createEnvironment(const MeshEnvironmentBuildInfo& infos) =0;

  /*!
   * \brief Créé un bloc.
   *
   * Créé un bloc avec les infos \a infos.
   *
   * La création d'un bloc ne peut avoir lieu que lors de l'initialisation,
   * (donc avant l'appel à endCreate()), mais après la création des milieux.
   */
  virtual IMeshBlock* createBlock(const MeshBlockBuildInfo& infos) =0;

  /*!
   * \brief Ajoute un milieu à un bloc existant.
   *
   * Ajoute le milieu \a env au bloc \a block.
   *
   * La modification d'un bloc ne peut avoir lieu que lors de l'initialisation,
   * (donc avant l'appel à endCreate()).
   *
   * \warning Cette méthode ne modifie pas le groupe block->cells() et c'est
   * donc à l'appelant d'ajouter au groupe les mailles du milieu si besoin.
   */
  virtual void addEnvironmentToBlock(IMeshBlock* block,IMeshEnvironment* env) =0;

  /*!
   * \brief Supprime un milieu à un bloc existant.
   *
   * Supprime le milieu \a env au bloc \a block.
   *
   * La modification d'un bloc ne peut avoir lieu que lors de l'initialisation,
   * (donc avant l'appel à endCreate()).
   *
   * \warning Cette méthode ne modifie pas le groupe block->cells() et c'est
   * donc à l'appelant d'ajouter au groupe les mailles du milieu si besoin.
   */
  virtual void removeEnvironmentToBlock(IMeshBlock* block,IMeshEnvironment* env) =0;

  /*!
   * \brief Indique qu'on a fini de créer les milieux.
   *
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été appelée.
   *
   * Si \a is_continue est vrai, recontruit pour chaque matériau et milieu
   * la liste de leurs mailles à partir des informations de reprise.
   */
  virtual void endCreate(bool is_continue=false) =0;

  /*!
   * \brief Recréé les infos des matériaux et milieux à partir des infos
   * de la protection.
   *
   * Cette méthode remplace le endCreate() et ne peut être utilisée qu'en reprise
   * et lors de l'initialisation.
   */
  virtual void recreateFromDump() =0;

  /*!
   * \brief Positionne la sauvegarde des valeurs entre deux modifications des
   * matériaux.
   *
   * Si actif, les valeurs des variables partielles sont conservées entre
   * deux modifications de la liste des matériaux.
   */
  virtual void setKeepValuesAfterChange(bool v) =0;
  
  //! Indique si les valeurs des variables sont conservées entre les modifications
  virtual bool isKeepValuesAfterChange() const =0;

  /*!
   * \brief Indique comment initialiser les nouvelles valeurs dans
   * les mailles matériaux et milieux.
   *
   * Si vrai, les nouvelles valeurs sont initialisées à zéro ou le vecteur
   * nul suivant le type de la donnée. Si faux, l'initialisation se fait avec
   * la valeur globale.
   */
  virtual void setDataInitialisationWithZero(bool v) =0;
  
  //! Indique comment initialiser les nouvelles valeurs dans les mailles matériaux et milieux.
  virtual bool isDataInitialisationWithZero() const =0;

  /*!
   * \brief Indique si les milieux et matériaux suivent les changements
   * de topologie dans le maillage.
   *
   * Cette méthode doit être apellée avant toute création de matériau.
   *
   * Si \a v vaut \a false, les milieux et les matériaux ne sont pas notifiés
   * des changements de la topologie du maillage. Dans ce cas, toutes les
   * données associées sont invalidées.
   */
  virtual void setMeshModificationNotified(bool v) =0;

  //! Indique si les milieux et matériaux suivent les changements de topologie dans le maillage.
  virtual bool isMeshModificationNotified() const =0;

  /*!
   * \brief Positionner les flags pour paramêtrer les modifications de matériaux/milieux.
   *
   * Les flags possibles sont une combinaison de eModificationFlags.
   *
   * Par exemple:
   \code
   IMeshMaterialMng* mm = ...;
   int flags = (int)eModificationFlags::GenericOptimize | (int)eModificationFlags::OptimizeMultiAddRemove;
   mm->setModificationFlags(flags);
   \endcode
   *
   * Cette méthode doit être activé avant l'appel à endCreate() pour être prise en compte.
   */
  virtual void setModificationFlags(int v) =0;

  //! Flags pour paramêtrer les modifications
  virtual int modificationFlags() const =0;

  /*!
   * \brief Positionne l'option indiquant si les variables scalaires
   * milieux sont allouées sur les matériaux.
   *
   * Si actif, alors les variables scalaires milieux sont tout de même allouées
   * aussi sur les matériaux. Cela permet de déclarer la même variable à la fois
   * comme une variable matériau et milieu (par exemple MaterialVariableCellReal et
   * EnvironmentVariableCellReal).
   *
   * Par défaut cette option n'est pas active.
   *
   * Cette méthode doit être activé avant l'appel à endCreate() pour être prise en compte.
   */
  virtual void setAllocateScalarEnvironmentVariableAsMaterial(bool v) =0;

  //! Indique si les variables scalaires milieux sont allouées sur les matériaux.
  virtual bool isAllocateScalarEnvironmentVariableAsMaterial() const =0;

  //! Nom du gestionnaire
  virtual String name() const =0;

  /*!
   * \ brief Nom du service utilisé pour compresser les données lors du forceRecompute().
   *
   * Si null (le défaut), aucune compression n'est effectuée.
   */
  virtual void setDataCompressorServiceName(const String& name) =0;

  //! virtual Nom du service utilisé pour compresser les données
  virtual String dataCompressorServiceName() const =0;

  //! Liste des matériaux
  virtual ConstArrayView<IMeshMaterial*> materials() const =0;

  //! Liste des matériaux vus comme composants
  virtual MeshComponentList materialsAsComponents() const =0;

  //! Liste des milieux
  virtual ConstArrayView<IMeshEnvironment*> environments() const =0;

  //! Liste des milieux vus comme composants
  virtual MeshComponentList environmentsAsComponents() const =0;

  /*!
   * \brief Liste de tous les composants.
   *
   * Cette liste est la concaténation de environmentsAsComponents() et
   * materialsAsComponents(). Elle n'est valide qu'une fois endCreate() appelé.
   */
  virtual MeshComponentList components() const =0;

  //! Liste des blocs
  virtual ConstArrayView<IMeshBlock*> blocks() const =0;

  /*!
   * \brief Retourne le milieux de nom \a name.
   *
   * Si aucune milieu de ce nom n'existe, retourne null si \a throw_exception est \a false
   * et lève une exception si \a throw_exception vaut \a true.
   */
  virtual IMeshEnvironment* findEnvironment(const String& name,bool throw_exception=true) =0;

  /*!
   * \brief Retourne le bloc de nom \a name.
   *
   * Si aucune bloc de ce nom n'existe, retourne null si \a throw_exception est \a false
   * et lève une exception si \a throw_exception vaut \a true.
   */
  virtual IMeshBlock* findBlock(const String& name,bool throw_exception=true) =0;

  /*!
   * \brief Remplit le tableau \a variables avec la liste des variables matériaux utilisés.
   *
   * La tableau \a variables est vidé avant l'appel.
   */
  virtual void fillWithUsedVariables(Array<IMeshMaterialVariable*>& variables) =0;

  //! Variable de nom \a name ou \a nullptr si aucune de ce nom existe.
  virtual IMeshMaterialVariable* findVariable(const String& name) =0;

  //! Variable aux matériaux associé à la variable global \a global_var (\a nullptr si aucune)
  virtual IMeshMaterialVariable* checkVariable(IVariable* global_var) =0;

  //! Ecrit les infos des matériaux et milieux sur le flot \a o
  virtual void dumpInfos(std::ostream& o) =0;

  //! Ecrit les infos de la maille \a cell sur le flot \a o
  virtual void dumpCellInfos(Cell cell,std::ostream& o) =0;

  //! Vérifie la validité des structures internes
  virtual void checkValid() =0;

  //! Vue sur les mailles milieux correspondant au groupe \a cells
  virtual AllEnvCellVectorView view(const CellGroup& cells) =0;

  //! Vue sur les mailles milieux correspondant au groupe \a cells
  virtual AllEnvCellVectorView view(CellVectorView cells) =0;

  //! Vue sur les mailles milieux correspondant aux mailles de numéro locaux cells_local_id
  virtual AllEnvCellVectorView view(SmallSpan<const Int32> cell_local_id) =0;

  //! Créée une instance pour convertir de 'Cell' en 'AllEnvCell'
  virtual CellToAllEnvCellConverter cellToAllEnvCellConverter() =0;

  /*!
   * \brief Force le recalcul des informations des matériaux.
   * 
   * Cette méthode permet de forcer le recalcul les informations sur les mailles
   * mixtes par exemple suite à un changement de maillage.
   * Il s'agit d'une méthode temporaire qui sera supprimée à terme.
   * Les valeurs mixtes sont invalidés après appel à cette méthode.
   */
  virtual void forceRecompute() =0;

  //! Verrou utilisé pour le multi-threading
  virtual Mutex* variableLock() =0;

  /*!
   * \brief Synchronise les mailles des matériaux.
   *
   * Cette méthode permet de synchroniser entre les sous-domaines les
   * mailles de chaque matériau. Elle est collective
   *
   * Lors de cet appel, le sous-domaine propriétaire de N mailles
   * envoie aux sous-domaines qui possède ces \a N mailles en tant que mailles
   * fantômes la liste des matériaux qu'il possède. Ces derniers sous-domaines
   * mettent à jour cette liste en ajoutant ou supprimant au besoin les
   * matériaux nécessaires.
   *
   * Après cet appel, il est garanti que
   * les mailles fantômes d'un sous-domaine ont bien la même liste de
   * matériaux et milieux que cells du sous-domaine qui est propriétaire
   * de ces mailles. Il est notamment possible de synchroniser des variables
   * via MeshMaterialVariableRef::synchronize().
   *
   * Retourne \a true si les matériaux de ce sous-domaine ont été modifiés suite
   * à la synchronisation, \a false sinon.
   */
  virtual bool synchronizeMaterialsInCells() =0;

  /*!
   * \brief Vérifie que les mailles des matériaux sont cohérentes entre
   * les sous-domaines.
   *
   * Cette méthode permet de vérifier que toutes les mailles fantômes
   * de notre sous-domaine ont bien la même liste de matériaux que
   * les mailles propres associées.
   *
   * En cas d'erreur, on affiche la liste des mailles qui ne sont pas
   * cohérentes et on lève une exception de type FatalErrorException.
   *
   * \a max_print indique en cas d'erreur le nombre maximal d'erreur à afficher.
   * S'il est négatif, on affiche toutes les mailles.
   */
  virtual void checkMaterialsInCells(Integer max_print=10) =0;

  //! Applique le fonctor \a functor sur l'ensemble des variables matériaux
  virtual void visitVariables(IFunctorWithArgumentT<IMeshMaterialVariable*>* functor) =0;

  /*!
   * \brief Compteur du nombre de modifications de la liste des matériaux
   * et des milieux.
   *
   * Ce compteur augmente à chaque fois que des matériaux sont ajoutés
   * ou supprimés. L'incrément n'est pas forcément constant.
   *
   * \note Actuellement, ce compteur n'est pas sauvegardé lors d'une
   * protection et vaudra donc 0 en reprise.
   */
  virtual Int64 timestamp() const =0;

  /*!
   * \brief Positionne la version de l'implémentation pour la synchronisation des
   * variables matériaux.
   */
  virtual void setSynchronizeVariableVersion(Integer version) =0;

  /*!
   * \brief Version de l'implémentation pour la synchronisation des
   * variables matériaux.
   */
  virtual Integer synchronizeVariableVersion() const =0;

  //! Vrai si on est en train de faire un échange de maillage avec gestion des matériaux.
  virtual bool isInMeshMaterialExchange() const =0;

  //! Interface de la fabrique de variables
  virtual IMeshMaterialVariableFactoryMng* variableFactoryMng() const =0;

  /*!
   * \brief Active ou désactive la construction et la mise à jour de la table de 
   * "connectivité" CellLocalId -> AllEnvCell pour les RUNCOMMAND
   *
   * On peut activer également par la variable d'environnement ARCANE_ALLENVCELL_FOR_RUNCOMMAND.
   * En option, on peut forcer la création de la table, ce qui peut être util lors d'un appel tardif
   * de cette méthode par rapport à celui du ForceRecompute()
   */
  virtual void enableCellToAllEnvCellForRunCommand(bool is_enable, bool force_create=false) =0;
  virtual bool isCellToAllEnvCellForRunCommand() const =0;

  /*!
   * \brief Indique si on utilise la valeur matériau ou milieu lorsqu'on transforme une maille
   * partielle en maille pure.
   *
   * Lors du passage d'une maille partielle en maille pure, il faut recopier la valeur
   * partielle dans la valeur globale. Par défaut, le comportement n'est pas le même
   * suivant que les optimisations sont actives ou non (\sa modificationFlags()).
   * Sans optimisation, c'est la valeur matériau qui est utilisée. Si l'optimisation
   * eModificationFlags::GenericOptimize est active, c'est la valeur milieu.
   *
   * Cette propriété, si elle vrai, permet d'utiliser la valeur matériau
   * dans tous les cas.
   */
  virtual void setUseMaterialValueWhenRemovingPartialValue(bool v) =0;
  virtual bool isUseMaterialValueWhenRemovingPartialValue() const =0;

 public:

  //!\internal
  class IFactory
  {
   public:
    virtual ~IFactory() = default;
    virtual Ref<IMeshMaterialMng> getTrueReference(const MeshHandle& mesh_handle,bool is_create) =0;
  };

 private:

  //!\internal
  static void _internalSetFactory(IFactory* f);

 public:

  //! API interne à %Arcane
  virtual IMeshMaterialMngInternal* _internalApi() const =0;

  /*!
   * \internal
   * \brief Synchronizeur pour les variables matériaux et milieux sur toutes les mailles.
   */
  virtual IMeshMaterialVariableSynchronizer* _allCellsMatEnvSynchronizer() = 0;

  /*!
   * \internal
   * \brief Synchronizeur pour les variables uniquement milieux sur toutes les mailles.
   */
  virtual IMeshMaterialVariableSynchronizer* _allCellsEnvOnlySynchronizer() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

