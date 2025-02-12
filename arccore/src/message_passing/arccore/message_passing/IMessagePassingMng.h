// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMessagePassingMng.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détruit l'instance \a p.
 *
 * L'instance \a p ne doit plus être utilisée après cet appel
 */
extern "C++" void ARCCORE_MESSAGEPASSING_EXPORT
mpDelete(IMessagePassingMng* p);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des échanges de messages.
 *
 * Ce gestionnaire ne fait pas grand chose en lui même et se contente de
 * déléguer les opérations via l'interface IDispatchers.
 *
 * Les instances de ces classes doivent être détruites via la méthode
 * mpDelete().
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng
{
  friend void ARCCORE_MESSAGEPASSING_EXPORT mpDelete(IMessagePassingMng*);
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  // TODO: Rendre obsolète fin 2022: [[deprecated("Use mpDelete() instead")]]
  virtual ~IMessagePassingMng() = default;

 public:

  //! Rang de cette instance dans le communicateur
  virtual Int32 commRank() const = 0;

  //! Nombre d'instance dans le communicateur
  virtual Int32 commSize() const = 0;

  //! Interface pour collecter les temps d'exécution (peut être nul)
  virtual ITimeMetricCollector* timeMetricCollector() const = 0;

  /*!
   * \brief Communicateur MPI associé à cette instance.
   *
   * Le communicateur n'est valide que si l'instance est associée à une
   * implémentation MPI.
   */
  virtual Communicator communicator() const;

 public:

  virtual IDispatchers* dispatchers() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
