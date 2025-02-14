// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StackTrace.h                                                (C) 2000-2025 */
/*                                                                           */
/* Informations d'une pile d'appel.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STACKTRACE_H
#define ARCCORE_BASE_STACKTRACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Conserve les adresses correspondantes à une pile d'appel.
 * Cette classe est interne et ne dois pas être utilisée en dehors d'Arccore.
 * \todo ajouter support pour windows.
 */
class StackFrame
{
 public:
  explicit StackFrame(intptr_t v) : m_address(v){}
  StackFrame() : m_address(0){}
 public:
  intptr_t address() const { return m_address; }
 private:
  intptr_t m_address;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Conserve une liste de taille fixe maximale de StackFrame.
 */
class FixedStackFrameArray
{
 public:
  static const int MAX_FRAME = 32;
  FixedStackFrameArray() : m_nb_frame(0){}
 public:
  ConstArrayView<StackFrame> view() const
  {
    return ConstArrayView<StackFrame>(m_nb_frame,m_addresses);
  }
  /*!
   * \brief Ajoute \a frame à la liste des frames. Si nbFrame() est supérieur
   * ou égal à MAX_FRAME, aucune opération n'est effectuée.
   */
  void addFrame(const StackFrame& frame)
  {
    if (m_nb_frame<MAX_FRAME){
      m_addresses[m_nb_frame] = frame;
      ++m_nb_frame;
    }
  }
  Integer nbFrame() const { return m_nb_frame; }
 private:
  //! Liste des adresses de la pile d'appel. Stocke au plus 32 appels.
  StackFrame m_addresses[MAX_FRAME];
  Integer m_nb_frame;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations sur la pile d'appel des fonctions.
 */
class ARCCORE_BASE_EXPORT StackTrace
{
 public:

  StackTrace() {}
  StackTrace(const FixedStackFrameArray& stack_frames) : m_stack_frames(stack_frames){}
  StackTrace(const String& msg) : m_stack_trace_string(msg){}
  StackTrace(const FixedStackFrameArray& stack_frames,const String& msg)
  : m_stack_frames(stack_frames), m_stack_trace_string(msg){}

 public:

  //! Chaîne de caractères indiquant la pile d'appel.
  const String& toString() const { return m_stack_trace_string; }

  //! Pile d'appel sous forme d'adresses.
  ConstArrayView<StackFrame> stackFrames() const { return m_stack_frames.view(); }

 private:

  FixedStackFrameArray m_stack_frames;
  String m_stack_trace_string;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateur d'écriture d'une StackTrace
ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o,const StackTrace&);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

