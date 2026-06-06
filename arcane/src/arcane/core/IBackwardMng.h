// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IBackwardMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface managing backward strategies.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IBACKWARDMNG_H
#define ARCANE_CORE_IBACKWARDMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface managing backward strategies.
 *
 * This interface is used by ITimeLoopMng to manage
 * backward rollback. The principle of backward rollback is to save at a given
 * iteration the values of the variables in order to be able to return to
 * this iteration, for example, in case of a calculation problem.
 *
 * It is possible to set a specific instance via
 * ITimeLoopMng::setBackwardMng();
 *
 * The sequence of operations, performed at the end of each iteration,
 * is managed by the ITimeLoopMng instance. It is as follows:
 *
 * \code
 * IBackwardMng bw = ...;
 * bw->beginAction();
 * if (bw->checkAndApplyRestore()){
 *   // Execution of restoration entry points.
 * }
 * bw->checkAndApplySave();
 * bw->endAction();
 * \endcode
 */
class ARCANE_CORE_EXPORT IBackwardMng
{
 public:

  // Actions to perform
  enum eAction
  {
    //! Save
    Save,
    //! Restore
    Restore
  };

 public:

  virtual ~IBackwardMng() = default;

 public:

  //! Initialization of the backward manager
  virtual void init() = 0;

  //! Indicates that the save/restore actions have started
  virtual void beginAction() = 0;

  /*!
   * \brief Checks and applies restoration if necessary.
   * \retval true if a restoration is performed.
   */
  virtual bool checkAndApplyRestore() = 0;

  /*!
   * \brief Checks and applies variable saving if necessary.
   * If \a is_forced is true, forces the save.
   * \retval true if a save is performed.
   */
  virtual bool checkAndApplySave(bool is_forced) = 0;

  //! Indicates that the save/restore actions are finished
  virtual void endAction() = 0;

  // Save period
  virtual void setSavePeriod(Integer n) = 0;

  // Retrieves the save period
  virtual Integer savePeriod() const = 0;

  /*!
   * \brief Signals that a backward rollback is desired.
   *
   * The backward rollback will occur when checkAndApplyRestore() is called.
   *
   * Generally, this method should not be called directly but
   * rather ITimeLoopMng::goBackward().
   *
   * From the call to this method until the effective action of
   * the backward rollback when calling checkAndApplyRestore(),
   * isBackwardEnabled() returns \a true.
   */
  virtual void goBackward() = 0;

  /*!
   * \brief Indicates if the backward rollback saves are locked.
   *
   * isLocked() is true if it is not possible to perform a
   * save. This is the case, for example, when a backward rollback has been performed at iteration
   * \a M to iteration N and we have not yet
   * returned to iteration \a M.
   */
  virtual bool isLocked() const = 0;

  /*!
   * \brief Indicates if a backward rollback is scheduled.
   * \sa goBackward().
   */
  virtual bool isBackwardEnabled() const = 0;

  /*!
   * \brief Deletes resources associated with the backward rollback.
   *
   * This method is called to deallocate resources
   * such as variable saves. This method is called
   * among other things before a load balancing because it will not be
   * possible to perform a backward rollback before this balancing.
   */
  virtual void clear() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
