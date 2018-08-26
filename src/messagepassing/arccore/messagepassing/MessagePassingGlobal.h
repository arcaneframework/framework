/*---------------------------------------------------------------------------*/
/* MessagePassingGlobal.h                                      (C) 2000-2018 */
/*                                                                           */
/* Définitions globales de la composante 'MessagePassing' de 'Arccore'.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEPASSINGGLOBAL_H
#define ARCCORE_MESSAGEPASSING_MESSAGEPASSINGGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_messagepassing)
#define ARCCORE_MESSAGEPASSING_EXPORT ARCCORE_EXPORT
#define ARCCORE_MESSAGEPASSING_EXTERN_TPL
#else
#define ARCCORE_MESSAGEPASSING_EXPORT ARCCORE_IMPORT
#define ARCCORE_MESSAGEPASSING_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{
//! Numéro correspondant à un rang nul
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

class Request;
class IStat;

/*!
 * \brief Types des réductions supportées.
 */
enum eReduceType
{
  ReduceMin, //!< Minimum des valeurs
  ReduceMax, //!< Maximum des valeurs
  ReduceSum  //!< Somme des valeurs
};

/*!
 * \brief Type d'attente.
 */
enum eWaitType
{
  WaitAll, //! Attend que tous les messages de la liste soient traités
  WaitSome,//! Attend que au moins un message de la liste soit traité
  WaitSomeNonBlocking //! Traite uniquement les messages qui peuvent l'être sans attendre.
};

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

