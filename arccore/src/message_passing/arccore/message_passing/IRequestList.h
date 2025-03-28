// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRequestList.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface d'une liste de requêtes de messages.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUESTLIST_H
#define ARCCORE_MESSAGEPASSING_REQUESTLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de requête de messages.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IRequestList
{
 public:

  virtual ~IRequestList() = default;

 public:

  //! Ajoute la requête \a r à la liste des requêtes
  virtual void add(Request r) =0;

  //! Ajoute la liste de requêtes \a rlist à la liste des requêtes
  virtual void add(Span<Request> rlist) =0;

  //! \a index-ième requête de la liste
  virtual Request request(Int32 index) const =0;

  //! Nombre de requêtes
  virtual Int32 size() const =0;

  //! Supprime toutes les requêtes de la liste
  virtual void clear() =0;

  /*!
   * \brief Attend ou test la complétion de une ou plusieurs requêtes.
   *
   * En retour, retourne le nombre de nouvelles requêtes terminées.
   * Il est ensuite possible de tester si une requête est terminée via la
   * méthode isRequestDone() ou de récupérer les indices des
   * requêtes terminées via doneRequestIndexes().
   *
   * \note Les requêtes terminées après un appel à wait() restent
   * dans la liste des requêtes. Il faut appeler la méthode
   * removeDoneRequests() si on souhaite les supprimer.
   */
  virtual Int32 wait(eWaitType wait_type) =0;

  //! Indique si la requête est terminée depuis le dernier appel à wait()
  virtual bool isRequestDone(Int32 index) const =0;

  /*!
   * \brief Supprime de la liste les requêtes terminées.
   *
   * Toutes les requêtes pour lesquelles isRequestDone() est vrai sont
   * supprimées de la liste des requêtes.
   * Après appel à cette méthode, on considère qu'il n'y a plus de
   * requêtes terminées. Par conséquent, doneRequestsIndexes() sera vide
   * et isRequestDone() retournera toujours \a false.
   */
  virtual void removeDoneRequests() =0;

  /*!
   * \brief Indices dans le tableaux des requêtes des requêtes terminées lors
   * du dernier appel à wait().
   */
  virtual ConstArrayView<Int32> doneRequestIndexes() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

