// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Profiler.h                                                  (C) 2000-2026 */
/*                                                                           */
/* Profiler class.                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_PROFILER_H
#define ARCCORE_ALINA_PROFILER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/AlinaGlobal.h"
#include "arccore/alina/ScopedStreamModifier.h"

#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Profiler class.
 *
 * Provides simple to use, hierarchical profile with nicely formatted output.
 */
class ARCCORE_ALINA_EXPORT Profiler
{
  static constexpr unsigned int SHIFT_WIDTH = 2;

 public:

  typedef double value_type;
  typedef double delta_type;

  //! Initialization.
  Profiler()
  : name("Profile")
  {
    init();
  }

  Profiler(const std::string& name)
  : name(name)
  {
    init();
  }

  /*!
   * \brief Starts measurement.
   *
   * \param name interval name.
   */
  void tic(const std::string& name);

  /*!
   * \brief Stops measurement.
   *
   * Returns delta in the measured value since the corresponding tic().
   */
  delta_type toc(const std::string& /*name*/ = "");

  static Profiler& globalProfiler();

  static void globalTic(const std::string& name);

  /*!
   * \brief Stops measurement.
   *
   * Returns delta in the measured value since the corresponding tic().
   */
  static delta_type globalToc(const std::string& /*name*/ = "");


  void reset();

  struct scoped_ticker
  {
    Profiler& prof;
    scoped_ticker(Profiler& prof)
    : prof(prof)
    {}
    ~scoped_ticker()
    {
      prof.toc();
    }
  };

  scoped_ticker scoped_tic(const std::string& name)
  {
    tic(name);
    return scoped_ticker(*this);
  }
  static scoped_ticker global_scoped_tic(const std::string& name)
  {
    return globalProfiler().scoped_tic(name);
  }

 private:

  struct profile_unit
  {
    profile_unit()
    : length(0)
    {}

    delta_type children_time() const
    {
      delta_type s = delta_type();
      for (typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end(); c++)
        s += c->second.length;
      return s;
    }

    size_t total_width(const std::string& name, int level) const
    {
      size_t w = name.size() + level;
      for (typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end(); c++)
        w = std::max(w, c->second.total_width(c->first, level + SHIFT_WIDTH));
      return w;
    }

    void print(std::ostream& out, const std::string& name,
               int level, delta_type total, size_t width) const
    {
      using namespace std;

      out << "[" << setw(level) << "";
      print_line(out, name, length, 100 * length / total, width - level);

      if (children.size()) {
        delta_type val = length - children_time();
        double perc = 100.0 * val / total;

        if (perc > 1e-1) {
          out << "[" << setw(level + 1) << "";
          print_line(out, "self", val, perc, width - level - 1);
        }
      }

      for (typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end(); c++)
        c->second.print(out, c->first, level + SHIFT_WIDTH, total, width);
    }

    void print_line(std::ostream& out, const std::string& name,
                    delta_type time, double perc, size_t width) const
    {
      using namespace std;

      out << name << ":"
          << setw(width - name.size()) << ""
          << setw(10)
          << fixed << setprecision(3) << time << " " << "s"
          << "] (" << fixed << setprecision(2) << setw(6) << perc << "%)"
          << endl;
    }

    value_type begin;
    delta_type length;

    std::map<std::string, profile_unit> children;
  };

  std::string name;
  profile_unit root;
  std::vector<profile_unit*> stack;

  void init();

  void print(std::ostream& out) const;

  /*!
   * \brief Sends formatted profiling data to an output stream.
   *
   * \param out  Output stream.
   * \param prof Profiler.
   */
  friend std::ostream& operator<<(std::ostream& out, const Profiler& prof)
  {
    out << std::endl;
    prof.print(out);
    return out << std::endl;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
