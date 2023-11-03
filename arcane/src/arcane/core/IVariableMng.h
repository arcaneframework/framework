// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableMng.h                                              (C) 2000-2023 */
/*                                                                           */
/* Interface du gestionnaire des variables.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEMNG_H
#define ARCANE_CORE_IVARIABLEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;
class IVariableFilter;
class VariableInfo;
class MeshVariable;
class IModule;
class IParallelMng;
class IDataReader;
class IDataWriter;
class IObservable;
class ICheckpointReader;
class CheckpointReadInfo;
class ICheckpointWriter;
class IPostProcessorWriter;
class VariableRef;
class IMesh;
class IVariableUtilities;
class VariableStatusChangedEventArgs;
class IVariableMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de variables.
 *
 * Ce gestionnaire contient la liste des variables déclarées dans le
 * sous-domaine associé \a subDomain(). Il maintient la liste des variables
 * et permet de les lire ou de les écrire.
 */
class IVariableMng
{
 public:

  virtual ~IVariableMng() = default; //!< Libère les ressources.

 public:

  //! Gestionnaire du sous-domaine
  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get 'ISubDomain' from another way")
  virtual ISubDomain* subDomain() =0;

  //! Gestionnaire de parallélisme associé
  virtual IParallelMng* parallelMng() const =0;

  //! Gestionnaire de messages
  virtual ITraceMng* traceMng() =0;

  /*!
   * \brief Vérifie une variable.
   *
   * Vérifie que la variable de nom \a name caractérisée par \a infos est valide
   * C'est le cas si et seulement si:
   * - aucune variable de nom \a infos.name() n'existe déjà.
   * - une variable de nom \a infos.name() existe et
   * son type et son genre correspondent \a infos.
   *
   * Si la variable n'est pas valide, une exception est lancée.
   * 
   * Cette opération est utilisée lorsqu'on souhaite créer une
   * nouvelle référence à une variable et permet de s'assurer qu'elle
   * sera valide.
   *
   * \exception ExBadVariableKindType si la variable de nom \a infos.name() existe
   * et que son type et le genre ne correspondent pas à ceux de \a infos.
   *
   * \return la variable de \a infos.name() si elle existe, 0 sinon
   */
  virtual IVariable* checkVariable(const VariableInfo& infos) =0;
  /*! \brief Génère un nom pour une variable temporaire.
   *
   * Pour assurer la cohérence de ce nom, il faut que tous les sous-domaines
   * appellent cette fonction.
   */
  virtual String generateTemporaryVariableName() =0;
  
  //! Affiche la liste des variables du gestionnaire lié à un module
  virtual void dumpList(std::ostream&,IModule*) =0;

  //! Affiche la liste de toutes les variables du gestionnaire
  virtual void dumpList(std::ostream&) =0;


  /*!
   * \brief Taille estimé pour exporter des variables.
   *
   Cette opération estime le nombre de méga octets que va générer
   l'exportation des variables \a vars. Si \a vars est vide, l'estimation
   porte sur toutes les variables référencées.
   
   L'estimation tient compte uniquement de la quantité mémoire utilisée
   par les variables et pas de l'écrivain utilisé.

   L'estimation est locale au sous-domaine. Pour obtenir la taille totale
   d'une exportation, il faut effectuer déterminer la taille par sous-domaine
   et faire la somme.

   Cette méthode est collective

   \todo utiliser des entiers 8 octets voir plus...
   */
  virtual Real exportSize(const VariableCollection& vars) =0;

  /*!
   * \brief Observable pour les variables en écriture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * avant d'écrire les variables (opération writeCheckpoint(),
   * writeVariables() ou writePostProcessing()).
   */
  virtual IObservable* writeObservable() =0;

  /*!
   * \brief Observable pour les variables en lecture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * après avoir lu les variables (opération readVariables() ou readCheckpoint()).
   */
  virtual IObservable* readObservable() =0;

  /*! \brief Ecrit les variables.
   *
   * Parcours l'ensemble des variables du gestionnaire et leur applique l'écrivain
   * \a writer. Si \a filter est non nul, il est appliqué à chaque variable et
   * une variable n'est écrite que si le filtre est vrai pour cette variable.
   *
   * Cette méthode est collective
   */
  virtual void writeVariables(IDataWriter* writer,IVariableFilter* filter=0) =0;

  /*!
   * \brief Exporte les variables.
   *
   * Exporte les variables de la liste \a vars. Si \a vars est
   * vide, exporte toutes les variables de la base qui sont utilisées.
   */
  virtual void writeVariables(IDataWriter* writer,const VariableCollection& vars) =0;

  /*!
   * \internal
   * \brief Ecrit les variables pour une protection.
   *
   * Utilise le service de protection \a writer pour écrire les variables.
   *
   * Cette méthode est collective.
   *
   * Cette méthode est interne à Arcane. En générel, l'écriture
   * d'une protection se fait via une instance de ICheckpointMng,
   * accessible via ISubDomain::checkpointMng().
   */
  virtual void writeCheckpoint(ICheckpointWriter* writer) =0;

  /*! \brief Ecrit les variables pour un post-traitement.
   *
   * Utilise le service de post-traitement \a writer pour écrire les variables.
   * L'appelant doit avoir positionner les champs de \a writer avant cet appel,
   * notamment la liste des variables à post-traiter. Cette méthode
   * appelle IPostProcessorWriter::notifyBeginWrite() avant l'écriture
   * et IPostProcessorWriter::notifyEndWriter() en fin.
   *
   * Cette méthode est collective.
   */
  virtual void writePostProcessing(IPostProcessorWriter* writer) =0;

  /*!
   *\brief Relit toutes les variables.
   *
   * Parcours l'ensemble des variables du gestionnaire et leur applique le lecteur
   * \a reader. Si \a filter est non nul, il est appliqué à chaque variable et
   * une variable n'est lue que si le filtre est vrai pour cette variable. Les
   * variables qui ne sont pas lues ne sont pas modifiées par cette opération.
   *
   * \deprecated Utiliser readVariable(IDataReader*)
   *
   * Cette méthode est collective.
   */
  virtual void readVariables(IDataReader* reader,IVariableFilter* filter=0) =0;

  /*!
   * \internal
   * \brief Relit toutes les variables d'une protection.
   *
   * Lit une protection avec le service \a reader sur l'ensemble
   * des variables.
   *
   * Cette méthode est collective.
   *
   * Cette méthode est interne à Arcane. En générel, la lecture
   * d'une protection se fait via une instance de ICheckpointMng,
   * accessible via ISubDomain::checkpointMng().
   */
  virtual void readCheckpoint(ICheckpointReader* reader) =0;

  /*!
   * \internal
   * \brief Relit toutes les variables d'une protection.
   *
   * Lit une protection avec les informations contenues
   * dans \a infos.
   *
   * Cette méthode est collective.
   *
   * Cette méthode est interne à Arcane. En générel, la lecture
   * d'une protection se fait via une instance de ICheckpointMng,
   * accessible via ISubDomain::checkpointMng().
   */
  virtual void readCheckpoint(const CheckpointReadInfo& infos) =0;

  //! Donne l'ensemble des variables du module \a i
  virtual void variables(VariableRefCollection v,IModule* i) =0;

  //! Liste des variables
  virtual VariableCollection variables() =0;

  //! Liste des variables utilisées
  virtual VariableCollection usedVariables() =0;
  
  //! Notifie au gestionnaire que l'état d'une variable a changé
  virtual void notifyUsedVariableChanged() =0;
  
  //! Retourne la variable de nom \a name ou 0 si aucune de se nom existe.
  virtual IVariable* findVariable(const String& name) =0;

  //! Retourne la variable du maillage de nom \a name ou 0 si aucune de se nom existe.
  virtual IVariable* findMeshVariable(IMesh* mesh,const String& name) =0;

  //! Retourne la variable de nom complet \a name ou 0 si aucune de se nom existe.
  virtual IVariable* findVariableFullyQualified(const String& name) =0;

  //! Ecrit les statistiques sur les variables sur le flot \a ostr
  virtual void dumpStats(std::ostream& ostr,bool is_verbose) =0;

  //! Ecrit les statistiques avec l'écrivain \a writer.
  virtual void dumpStatsJSON(JSONWriter& writer) =0;

  //! Interface des fonctions utilitaires associées
  virtual IVariableUtilities* utilities() const =0;

  //! Interface du gestionnaire de synchronisation des variables.
  virtual IVariableSynchronizerMng* synchronizerMng() const =0;

 public:

  //! \name Evènements
  //@{
  //! Evènement envoyé lorsqu'une variable est créée
  virtual EventObservable<const VariableStatusChangedEventArgs&>& onVariableAdded() =0;

  //! Evènement envoyé lorsqu'une variable est détruite
  virtual EventObservable<const VariableStatusChangedEventArgs&>& onVariableRemoved() =0;
  //@}

 public:

  /*!
   * \brief Construit les membres de l'instance.
   *
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée. Cette méthode doit être appelée avant initialize().
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void build() =0;

  /*!
   * \brief Initialise l'instance.
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée.
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void initialize() =0;

  //! Supprime et détruit les variables gérées par ce gestionnaire
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void removeAllVariables() =0;

  //! Détache les variables associées au maillage \a mesh.
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void detachMeshVariables(IMesh* mesh) =0;


  /*!
   * \brief Ajoute une référence à une variable.
   *
   * Ajoute la référence \a var au gestionnaire.
   *
   * \pre var != 0
   * \pre var ne doit pas déjà être référencée.
   * \return l'implémentation associée à \a var.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addVariableRef(VariableRef* var) =0;

  /*!
   * \brief Supprime une référence à une variable.
   *
   * Supprime la référence \a var du gestionnaire.
   *
   * Si \a var n'est pas référencée par le gestionnaire, rien n'est effectué.
   * \pre var != 0
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void removeVariableRef(VariableRef* var) =0;

  /*!
   * \brief Ajoute une variable.
   *
   * Ajoute la variable \a var.
   *
   * La validité de la variable n'est pas effectuée (void checkVariable()).
   *
   * \pre var != 0
   * \pre var ne doit pas déjà être référencée.
   * \return l'implémentation associée à \a var.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addVariable(IVariable* var) =0;

  /*!
   * \brief Supprime une variable.
   *
   * Supprime la variable \a var.
   *
   * Après appel à cette méthode, la variable ne doit plus être utilisée.
   *
   * \pre var != 0
   * \pre var doit avoir une seule référence.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void removeVariable(IVariable* var) =0;

  /*!
   * \brief Initialise les variables.
   *
   * Parcours la liste des variables et les initialisent.
   * Seules les variables d'un module utilisé sont initialisées.
   *
   * \param is_continue \a true vrai si on est en reprise.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void initializeVariables(bool is_continue) =0;

 public:

  /*!
   * \internal
   * Fonction interne temporaire pour récupérer le sous-domaine.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual ISubDomain* _internalSubDomain() const =0;

 public:

  //! API interne à Arcane
  virtual IVariableMngInternal* _internalApi() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
