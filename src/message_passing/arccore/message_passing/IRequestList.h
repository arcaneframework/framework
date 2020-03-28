// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* IRequestList.h                                              (C) 2000-2020 */
/*                                                                           */
/* Interface d'une liste de requêtes de messages.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUESTLIST_H
#define ARCCORE_MESSAGEPASSING_REQUESTLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
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

  //! Ajoute la requête \a r à la liste
  virtual void addRequest(Request r) =0;

  //! Nombre de requêtes
  virtual Integer nbRequest() const =0;

  /*!
   * \brief Attend ou test la complétion de une ou plusieurs requêtes.
   *
   * Cette méthode appelle removeDoneRequest() avant d'attendre la
   * complétion. En retour, retourne le nombre de requêtes terminées.
   * Il est possible de tester si une requête est terminée via la
   * méthode isRequestDone();
   */
  virtual Integer wait(eWaitType wait_type) =0;

  //! Indique si la requête est terminée
  virtual bool isRequestDone(Integer index) const =0;

  //! \a index-ième requête de la liste
  virtual Request request(Integer index) const =0;

  //! Supprime de la liste les requêtes terminées
  virtual void removeDoneRequests() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

