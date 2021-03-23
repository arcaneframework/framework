// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPerformanceCounterService.h                                (C) 2000-2016 */
/*                                                                           */
/* Interface d'un service d'accès aux compteurs de performance.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IPERFORMANCECOUNTERSERVICE_H
#define ARCANE_UTILS_IPERFORMANCECOUNTERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimerMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service d'accès aux compteurs de performance.
 */
class ARCANE_UTILS_EXPORT IPerformanceCounterService
{
 public:

  virtual ~IPerformanceCounterService() {}

 public:

  //! Initialise le service.
  virtual void initialize() =0;

  /*!
   * \brief Débute le suivi des compteurs de performance.
   * \pre isStarted()==false.
   * \post isStarted()==true.
   */
  virtual void start() =0;

  /*!
   * \brief Arrête le suivi des compteurs de performance.
   * \pre isStarted()==true.
   * \post isStarted()==false.
   */
  virtual void stop() =0;

  //! Indique si le service a démarré (start() a été appelé)
  virtual bool isStarted() const =0;

  /*!
   * \brief Récupère les valeurs actuelles des compteurs.
   *
   * Cette méthode ne doit être appelée que si isStarted() est vrai.
   *
   * Si \a do_substract vaut \a false, remplit \a counters avec les
   * valeurs actuelles des compteurs. Si \a do_substract vaut \a true,
   * remplit counters avec la différence entre les valeurs actuelles et celles
   * de \a counters lors de l'appel.
   *
   \code
   * Int64ArrayView counters = ...;
   * IPerformanceCounterService* p = ...;
   * p->getCounters(counters,false);
   * ... // Operation.
   * p->getCounters(counters,true);
   * info() << "Nb cycle=" << counters[0].
   \endcode
   *
   * Le compteur d'indice 0 est toujours le nombre de cycle. \a counters
   * doit valoir assez d'élémentts pour renseigner au moins 8 compteurs.
   *
   * \retval le nombre de compteurs renseignés.
   * \pre isStarted()==true
   */
  virtual Integer getCounters(Int64ArrayView counters,bool do_substract) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
