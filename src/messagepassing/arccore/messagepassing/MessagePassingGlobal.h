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
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

