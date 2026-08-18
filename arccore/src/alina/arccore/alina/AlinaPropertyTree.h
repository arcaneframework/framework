// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaPropertyTree.h                                           (C) 2000-2026 */
/*                                                                           */
/* Minimal hierarchical property tree used to store Alina parameters.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_ALINAPROPERTYTREE_H
#define ARCCORE_ALINA_ALINAPROPERTYTREE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <cstddef>
#include <list>
#include <string>
#include <utility>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{
namespace detail
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Minimal hierarchical property tree with string leaf values.
 *
 * This class replaces the previous use of 'boost::property_tree::ptree'.
 * It only implements the subset of the Boost API that is needed by Alina:
 * path-based access using '.' as separator, insertion-ordered children and
 * string values for leaves. A subtree is itself a 'PropertyTreeImpl' (like
 * the Boost class), so a 'PropertyTree' can alias any node of the tree.
 */
class PropertyTreeImpl
{
 public:

  std::string data;
  bool has_data = false;
  std::list<std::pair<std::string, PropertyTreeImpl>> children;

 public:

  bool empty() const { return !has_data && children.empty(); }

 public:

  //! First direct child with name \a name or nullptr if not found.
  PropertyTreeImpl* findChild(const std::string& name)
  {
    for (auto& c : children)
      if (c.first == name)
        return &c.second;
    return nullptr;
  }
  const PropertyTreeImpl* findChild(const std::string& name) const
  {
    for (const auto& c : children)
      if (c.first == name)
        return &c.second;
    return nullptr;
  }

  //! Number of direct children with name \a name.
  size_t count(const std::string& name) const
  {
    size_t n = 0;
    for (const auto& c : children)
      if (c.first == name)
        ++n;
    return n;
  }

  //! Remove the first direct child with name \a name.
  bool eraseChild(const std::string& name)
  {
    auto it = children.begin();
    while (it != children.end()) {
      if (it->first == name) {
        children.erase(it);
        return true;
      }
      ++it;
    }
    return false;
  }

 public:

  //! Node at the '.'-separated path \a path or nullptr if not found.
  //! An empty path corresponds to this node.
  PropertyTreeImpl* findNode(const std::string& path)
  {
    if (path.empty())
      return this;
    PropertyTreeImpl* n = this;
    for (const auto& part : splitPath(path)) {
      n = n->findChild(part);
      if (!n)
        return nullptr;
    }
    return n;
  }
  const PropertyTreeImpl* findNode(const std::string& path) const
  {
    if (path.empty())
      return this;
    const PropertyTreeImpl* n = this;
    for (const auto& part : splitPath(path)) {
      n = n->findChild(part);
      if (!n)
        return nullptr;
    }
    return n;
  }

  //! Leaf value at the '.'-separated path \a path or nullptr if not found.
  const std::string* getValue(const std::string& path) const
  {
    const PropertyTreeImpl* n = findNode(path);
    if (n && n->has_data)
      return &n->data;
    return nullptr;
  }

 public:

  //! Set the leaf value at the '.'-separated path \a path, creating
  //! intermediate nodes if needed.
  void put(const std::string& path, const std::string& value)
  {
    PropertyTreeImpl* n = createNode(path);
    n->data = value;
    n->has_data = true;
  }

  //! Append \a subtree as a direct child whose name is the last component
  //! of the '.'-separated path \a path, creating intermediate nodes if needed.
  void addChild(const std::string& path, const PropertyTreeImpl& subtree)
  {
    std::vector<std::string> parts = splitPath(path);
    if (parts.empty())
      return;
    std::string parent_path;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      if (!parent_path.empty())
        parent_path += '.';
      parent_path += parts[i];
    }
    PropertyTreeImpl* parent = createNode(parent_path);
    parent->children.emplace_back(std::move(parts.back()), subtree);
  }

 private:

  static std::vector<std::string> splitPath(const std::string& path)
  {
    std::vector<std::string> parts;
    std::string current;
    for (char c : path) {
      if (c == '.') {
        parts.push_back(current);
        current.clear();
      }
      else {
        current += c;
      }
    }
    parts.push_back(current);
    return parts;
  }

  //! Node at the '.'-separated path \a path, creating intermediate nodes.
  PropertyTreeImpl* createNode(const std::string& path)
  {
    PropertyTreeImpl* n = this;
    for (const auto& part : splitPath(path)) {
      PropertyTreeImpl* next = n->findChild(part);
      if (!next)
        next = &n->children.emplace_back(part, PropertyTreeImpl()).second;
      n = next;
    }
    return n;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace detail
} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
