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
