// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien
{
namespace Exception
{
  class BaseException : public std::exception
  {
   public:
    BaseException(const char* type, const char* msg, int line)
    {
      std::ostringstream oss;
      oss << "Error type " << type << " line " << line << " : "
          << msg;
      this->msg = oss.str();
    }

    BaseException(const char* type, const char* msg)
    {
      std::ostringstream oss;
      oss << "Error type " << type << " : " << msg;
      this->msg = oss.str();
    }

    virtual ~BaseException() noexcept(true)
    {
    }

    virtual const char* what() const noexcept(true)
    {
      return this->msg.c_str();
    }

   private:
    std::string msg;
  };

  class NumericException : public BaseException
  {
   public:
    NumericException(std::string const& msg, int line)
    : BaseException("Numeric", msg.c_str(), line)
    {}
  };
} // namespace Exception
} // namespace Alien
