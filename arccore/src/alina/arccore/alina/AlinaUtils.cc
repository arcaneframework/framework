// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaUtils.cc                                               (C) 2000-2026 */
/*                                                                           */
/* Classes utilitaires.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/CSRMatrixView.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using BoostPTree = boost::property_tree::ptree;

namespace detail
{
  const BoostPTree& empty_ptree()
  {
    static const BoostPTree p;
    return p;
  }

  // To access PropertyTree::m_property_tree as 'boost::property_tree::ptree'
  class PropertyWrapper
  {
   public:

    static const BoostPTree& toBoostPTree(const PropertyTree& p)
    {
      return *(static_cast<const BoostPTree*>(p.m_property_tree));
    }
    static BoostPTree& toBoostPTree(PropertyTree& p)
    {
      return *(static_cast<BoostPTree*>(p.m_property_tree));
    }
  };

} // namespace detail

namespace
{
  const BoostPTree& toBoostPTree(const PropertyTree& p)
  {
    return detail::PropertyWrapper::toBoostPTree(p);
  }
  const BoostPTree& toBoostPTree(const PropertyTree* p)
  {
    return detail::PropertyWrapper::toBoostPTree(*p);
  }
  BoostPTree& toBoostPTree(PropertyTree* p)
  {
    return detail::PropertyWrapper::toBoostPTree(*p);
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree::
PropertyTree()
: m_property_tree(new BoostPTree())
, m_is_own(true)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree::
PropertyTree(const PropertyTree& rhs)
{
  if (rhs.m_is_own) {
    m_property_tree = new BoostPTree(toBoostPTree(rhs));
    m_is_own = true;
  }
  else {
    m_property_tree = rhs.m_property_tree;
    m_is_own = false;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*PropertyTree::
PropertyTree(const BoostPTree& x)
: m_property_tree(new BoostPTree(x))
, m_is_own(true)
{}*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree::
~PropertyTree()
{
  if (m_is_own)
    delete &toBoostPTree(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree PropertyTree::
get_child_empty(const std::string& path) const
{
  const BoostPTree& child = toBoostPTree(this).get_child(path, detail::empty_ptree());
  PropertyTree p;
  p.m_property_tree = const_cast<BoostPTree*>(&child);
  p.m_is_own = false;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PropertyTree::
erase(const char* name)
{
  return toBoostPTree(this).erase(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

size_t PropertyTree::
count(const char* name) const
{
  return toBoostPTree(this).count(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
read_json(const std::string& filename)
{
  BoostPTree& p = toBoostPTree(this);
  boost::property_tree::json_parser::read_json(filename, p);
}

Int32 PropertyTree::get(const char* param_type, Int32 default_value) const
{
  return toBoostPTree(this).get(param_type, default_value);
}
Int64 PropertyTree::get(const char* param_type, Int64 default_value) const
{
  return toBoostPTree(this).get(param_type, default_value);
}
double PropertyTree::get(const char* param_type, double default_value) const
{
  return toBoostPTree(this).get(param_type, default_value);
}
double* PropertyTree::get(const char* param_type, double* default_value) const
{
  return toBoostPTree(this).get(param_type, default_value);
}
void* PropertyTree::get(const char* param_type, void* default_value) const
{
  return toBoostPTree(this).get(param_type, default_value);
}
std::string PropertyTree::get(const char* param_type, const std::string& default_value) const
{
  return toBoostPTree(this).get(param_type, default_value);
}

void PropertyTree::put(const std::string& path, Int32 value)
{
  toBoostPTree(this).put(path, value);
}
void PropertyTree::put(const std::string& path, Int64 value)
{
  toBoostPTree(this).put(path, value);
}
void PropertyTree::put(const std::string& path, double value)
{
  toBoostPTree(this).put(path, value);
}
void PropertyTree::put(const std::string& path, const std::string& value)
{
  toBoostPTree(this).put(path, value);
}
void PropertyTree::put(const std::string& path, double* value)
{
  toBoostPTree(this).put(path, value);
}
void PropertyTree::put(const std::string& path, void* value)
{
  toBoostPTree(this).put(path, value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
_addChild(const std::string& path, const char* name,
          const PropertyTree& obj)
{
  auto& p = toBoostPTree(this);
  p.add_child(std::string(path) + name, toBoostPTree(obj));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
check_params(const std::set<std::string>& names) const
{
  const auto& p = toBoostPTree(this);
  bool has_error = false;
  for (const auto& n : names) {
    if (!p.count(n)) {
      ARCCORE_ALINA_PARAM_MISSING(n);
    }
  }
  for (const auto& v : p) {
    if (!names.count(v.first)) {
      std::cerr << "WARNING: unknown parameter " << v.first << "\n";
      has_error = true;
    }
  }
  if (has_error)
    ARCANE_FATAL("Invalid parameters");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
check_params(const std::set<std::string>& names,
             const std::set<std::string>& opt_names) const
{
  const auto& p = toBoostPTree(this);
  bool has_error = false;

  for (const auto& n : names) {
    if (!p.count(n)) {
      ARCCORE_ALINA_PARAM_MISSING(n);
    }
  }
  for (const auto& n : opt_names) {
    if (!p.count(n)) {
      ARCCORE_ALINA_PARAM_MISSING(n);
    }
  }
  for (const auto& v : p) {
    if (!names.count(v.first) && !opt_names.count(v.first)) {
      std::cerr << "WARNING: unknown parameter " << v.first << "\n";
      has_error = true;
    }
  }
  if (has_error)
    ARCANE_FATAL("Invalid parameters");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
putKeyValue(const std::string& param)
{
  auto& p = toBoostPTree(this);
  size_t eq_pos = param.find('=');
  if (eq_pos == std::string::npos)
    ARCANE_FATAL("param in put() should have \"key=value\" format (param='{0}')", param);
  p.put(param.substr(0, eq_pos), param.substr(eq_pos + 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

detail::empty_params::
empty_params(const PropertyTree& ap)
{
  const auto& p = toBoostPTree(ap);
  for (const auto& v : p) {
    std::cerr << "Alina: unknown parameter " << v.first << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o, const PropertyTree& obj)
{
  const auto& p = toBoostPTree(obj);
  std::ostringstream ostr;
  boost::property_tree::json_parser::write_json(ostr, p);
  o << ostr.str();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
