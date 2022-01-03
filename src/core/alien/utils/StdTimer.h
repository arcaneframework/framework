/*
 * StdTimer.h
 *
 *  Created on: Dec 1, 2021
 *      Author: gratienj
 */
#pragma once
#include <string>
#include <map>
#include <chrono>

namespace Alien
{
class StdTimer
{
 public:
  class Sentry
  {
   public:
    Sentry(StdTimer& parent, std::string phase)
    : m_parent(parent)
    , m_phase(phase)
    {
      m_start = std::chrono::steady_clock::now();
    }

    virtual ~Sentry()
    {
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - m_start);

      m_parent.add(m_phase, time_span.count());
    }

   private:
    StdTimer& m_parent;
    std::string m_phase;
    std::chrono::steady_clock::time_point m_start;
  };

  StdTimer() {}
  virtual ~StdTimer() {}

  void add(std::string const& phase, double value)
  {
    auto iter = m_counters.find(phase);
    if (iter == m_counters.end())
      m_counters[phase] = value;
    else
      iter->second += value;
  }

  void printInfo(const std::string& msg) const
  {
    std::cout << "================================" << std::endl;
    std::cout << "PERF INFO : " << msg << std::endl;
    for (auto const& iter : m_counters) {
      std::cout << iter.first << ":" << iter.second << std::endl;
    }
    std::cout << "================================" << std::endl;
  }

  void printInfo(std::ostream& out, const std::string& msg) const
  {
    out << msg << std::endl;
    out << "================================" << std::endl;
    out << "PERF INFO : " << std::endl;
    for (auto const& iter : m_counters) {
      out << iter.first << ":" << iter.second << std::endl;
    }
    out << "================================" << std::endl;
  }

 private:
  std::map<std::string, double> m_counters;
};
} // namespace Alien
