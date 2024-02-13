// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*
 * SYCLEnv.h
 *
 *  Created on: Nov 26, 2021
 *      Author: gratienj
 */

#pragma once

namespace Alien
{
namespace SYCLInternal
{
  struct EnvInternal;
}

class ALIEN_EXPORT SYCLEnv
{
 public:
  static SYCLEnv* m_instance;
  static SYCLEnv* instance();

  SYCLEnv();

  virtual ~SYCLEnv();

  SYCLInternal::EnvInternal* internal()
  {
    return m_internal;
  }

  std::size_t maxNumGroups();

  std::size_t maxWorkGroupSize();

  std::size_t maxNumThreads();

 private:
  SYCLInternal::EnvInternal* m_internal = nullptr;
};
} // namespace Alien
