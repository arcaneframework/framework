// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelSuperMng.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface du superviseur du parallélisme.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELSUPERMNG_H
#define ARCANE_CORE_IPARALLELSUPERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IApplication;
class IParallelMng;
class IThreadMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe abstraite du superviseur de parallélisme.
 */
class ARCANE_CORE_EXPORT IParallelSuperMng
{
 public:

  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;

 public:
	
  virtual ~IParallelSuperMng() {} //!< Libère les ressources.

 public:

  /*!
   * \brief Construit les membres l'instance.
   *
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée. Cette méthode doit être appelée avant initialize().
   *
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void build() =0;

  /*!
   * \brief Initialise l'instance.
   *
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée.
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void initialize() =0;

 public:

  //! Retourne le gestionnaire principal.
  virtual IApplication* application() const =0;

  //! Gestionnaire de thread.
  virtual IThreadMng* threadMng() const =0;
	
  //! Retourne \a true si l'exécution est parallèle
  virtual bool isParallel() const =0;

  //! Retourne le numéro du process (compris entre 0 et nbProcess()-1)
  virtual Int32 commRank() const =0;

  //! Retourne le nombre total de process utilisés
  virtual Int32 commSize() const =0;

  //! Rang de cette instance pour les traces.
  virtual Int32 traceRank() const =0;

  /*!
   * \brief Adresse du communicateur MPI associé à ce gestionnaire.
   *
   * Le communicateur n'est valide que si on utilise MPI. Sinon, l'adresse
   * retournée est 0. La valeur retournée est de type (MPI_Comm*).
   */

  virtual void* getMPICommunicator() =0;

  /*!
   * \brief Communicateur MPI associé à ce gestionnaire
   *
   * \sa IParallelMng::communicator()
   */
  virtual Parallel::Communicator communicator() const =0;

  /*!
   * \internal
   * \brief Créé un gestionnaire de parallélisme pour l'ensemble des coeurs alloués.
   *
   * Cette opération est collective.
   *
   * Cette méthode ne doit être appelée qu'une seule fois lors de l'initialisation.
   *
   * \a local_rank est le rang local de l'appelant dans la liste des rangs.
   * En mode pure MPI, ce rang est toujours 0 car il n'y a qu'un seul
   * thread. En mode Thread ou Thread/MPI, il s'agit du rang du thread utilisé
   * lors de la création.
   *
   * Le gestionnaire retourné reste la propriété de cette instance et 
   * ne doit pas être détruit.
   *
   * A usage interne uniquement.
   */
  virtual Ref<IParallelMng> internalCreateWorldParallelMng(Int32 local_rank) =0;

  /*!
   * \brief Nombre de sous-domaines à créér localement.
   * - 1 si séquentiel.
   * - 1 si MPI pur
   * - n si THREAD ou THREAD/MPI
   */
  virtual Int32 nbLocalSubDomain() =0;

  /*!
   * \brief Tente de faire un abort.
   *
   * Cette méthode est appelée lorsqu'une exception a été généré et que le
   * cas en cours d'exécution doit s'interrompre. Elle permet d'effectuer les
   * opérations de nettoyage du gestionnaire si besoin est.
   */
  virtual void tryAbort() =0;

  //! \a true si l'instance est un gestionnaire maître des entrées/sorties.
  virtual bool isMasterIO() const =0;

  /*!
    \brief Rang de l'instance gérant les entrées/sorties (pour laquelle isMasterIO() est vrai)
    *
    * Dans l'implémentation actuelle, il s'agit toujours du processeur de rang 0.
    */
  virtual Int32 masterIORank() const =0;

  /*!
   * \brief Gestionnaire de parallèlisme pour l'ensemble des ressources allouées.
   */
  //virtual IParallelMng* worldParallelMng() const =0;

  //! Effectue une barière
  virtual void barrier() =0;

 public:

  //! @name opérations de broadcast
  //@{
  /*!
   * \brief Envoie un tableau de valeurs sur tous les processus
   * Cette opération synchronise le tableau de valeur \a send_buf sur tous
   * les processus. Le tableau utilisé est celui du processus dont
   * l'identifiant (processId()) est \a process_id.
   * Tous les processus doivent appelés cette méthode avec
   * le même paramètre \a process_id et avoir un tableau \a send_buf
   * contenant le même nombre d'éléments.
   */
  virtual void broadcast(ByteArrayView send_buf,Integer process_id) =0;
  virtual void broadcast(Int32ArrayView send_buf,Integer process_id) =0;
  virtual void broadcast(Int64ArrayView send_buf,Integer process_id) =0;
  virtual void broadcast(RealArrayView send_buf,Integer process_id) =0;
  //@}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
