// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DefaultBackwardMng.h                                        (C) 2000-2016 */
/*                                                                           */
/* Default implementation of a backward strategy.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MAIN_DEFAULTBACKWARDMNG_H
#define ARCANE_MAIN_DEFAULTBACKWARDMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IBackwardMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableFilter;
class IDataReaderWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Default implementation of a backward strategy.
 */
class ARCANE_IMPL_EXPORT DefaultBackwardMng
: public IBackwardMng
{
 private:

  enum eSequence
  {
    //! Save
    SEQSave,
    //! Forced save
    SEQForceSave,
    //! Restore
    SEQRestore,
    //! Lock
    SEQLock,
    //! Nothing
    SEQNothing
  };

 public:

  DefaultBackwardMng(ITraceMng* trace, ISubDomain* sub_domain);
  ~DefaultBackwardMng();

  void init() override {}

  void beginAction() override;

  void endAction() override;

  void setSavePeriod(Integer n) override { m_period = n; }
  Integer savePeriod() const override { return m_period; }

  void goBackward() override;

  bool isLocked() const override { return m_sequence == SEQLock; }

  bool isBackwardEnabled() const override { return m_sequence == SEQRestore; }

  void clear() override;

  virtual bool checkAndApplyRestore() override;
  virtual bool checkAndApplySave(bool is_forced) override;

 private:

  void _restore();

  void _save();

 private:

  ITraceMng* m_trace;
  ISubDomain* m_sub_domain;
  IVariableFilter* m_filter;
  IDataReaderWriter* m_data_io;

  //! Time of the last requested backward action
  Real m_backward_time;

  //! Period between two saves for backward tracking
  Integer m_period;

  //! First save
  bool m_first_save;

  //! Actions allowed?
  bool m_action_refused;

  //! Sequence
  eSequence m_sequence;

 private:

  void _checkValidAction();
  void _checkSave(bool is_forced);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
