// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaSamplesCommon.h                                        (C) 2000-2026 */
/*                                                                           */
/* Utilitary functions used by all samples.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_ALINASAMPLESCOMMON_H
#define ARCCORE_ALINA_ALINASAMPLESCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/AlinaGlobal.h"
#include "arccore/trace/TraceGlobal.h"
#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"
#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_alina
#define ARCCORE_ALINA_SAMPLES_EXPORT ARCANE_EXPORT
#else
#define ARCCORE_ALINA_SAMPLES_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

class SampleMainContext;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Prototype for main functor
typedef int (*MainFunction)(const SampleMainContext& ctx, int argc, char* argv[]);

class ARCCORE_ALINA_SAMPLES_EXPORT SampleMainContext
{
 private:

  SampleMainContext(ITraceMng* tm, Accelerator::IAcceleratorMng* accelerator_mng,
                    MessagePassing::IMessagePassingMng* message_passing_mng)
  : m_trace_mng(tm)
  , m_accelerator_mng(accelerator_mng)
  , m_message_passing_mng(message_passing_mng)
  {}

 public:

  static int execMain(MainFunction f, int argc, char* argv[]);

 public:

  ITraceMng* traceMng() const { return m_trace_mng; }
  Accelerator::IAcceleratorMng* acceleratorMng() const { return m_accelerator_mng; }
  MessagePassing::IMessagePassingMng* messagePassingMng() const { return m_message_passing_mng; }

 private:

  ITraceMng* m_trace_mng = nullptr;
  Accelerator::IAcceleratorMng* m_accelerator_mng = nullptr;
  MessagePassing::IMessagePassingMng* m_message_passing_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
