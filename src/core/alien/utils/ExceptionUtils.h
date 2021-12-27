/*
 * ExceptionUtils.h
 *
 *  Created on: Dec 2, 2021
 *      Author: gratienj
 */

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

    virtual ~BaseException() throw()
    {
    }

    virtual const char* what() const throw()
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
