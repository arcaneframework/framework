// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterCaseOption.h                                       (C) 2000-2025 */
/*                                                                           */
/* Class allowing querying parameters to determine if dataset options        */
/* should be modified by them.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_PARAMETERCASEOPTION_H
#define ARCANE_UTILS_INTERNAL_PARAMETERCASEOPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterOptionElementsCollection;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class representing the set of parameters that can modify
 * the dataset options.
 */
class ARCANE_UTILS_EXPORT ParameterCaseOption
{

 public:

  ParameterCaseOption(ParameterOptionElementsCollection* parameter_options, const String& lang);

 public:

  /*!
   * \brief Method allowing retrieval of an option's value.
   *
   * The option address is formatted as follows:
   * xpath_before_index[index]/xpath_after_index
   *
   * xpath_before_index must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by the one passed as a parameter).
   *
   * xpath_after_index must be in the following format:
   * ddd/eee
   * - no "/" at the beginning or the end.
   *
   * The indices are XML indices and these indices start at 1.
   *
   * \param xpath_before_index The address before the index.
   * \param xpath_after_index The address after the index.
   * \param index The index to place between the two parts of the address.
   * \return The value if found, otherwise null string.
   */
  String getParameterOrNull(const String& xpath_before_index, const String& xpath_after_index, Integer index) const;

  /*!
   * \brief Method allowing retrieval of an option's value.
   *
   * The option address is formatted as follows:
   * xpath_before_index[index]
   *
   * xpath_before_index must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by the one passed as a parameter).
   *
   * If the parameter allow_elems_after_index is enabled, addresses of the form:
   * xpath_before_index[index]/aaa/bbb
   * will also be searched.
   *
   * The indices are XML indices and these indices start at 1.
   *
   * \param xpath_before_index The address before the index.
   * \param index The index to place after the address.
   * \param allow_elems_after_index Should elements after the index be checked?
   * \return The value if found, otherwise null string.
   */
  String getParameterOrNull(const String& xpath_before_index, Integer index, bool allow_elems_after_index) const;

  /*!
   * \brief Method allowing retrieval of an option's value.
   *
   * The address must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end.
   *
   * The indices are XML indices and these indices start at 1.
   *
   * \param full_xpath The address to search for.
   * \return The value if found, otherwise null string.
   */
  String getParameterOrNull(const String& full_xpath) const;

  /*!
   * \brief Method allowing checking if an option is present.
   *
   * The address must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end.
   *
   * The indices are XML indices and these indices start at 1.
   *
   * \param full_xpath The address to search for.
   * \return true if the address is found in the list.
   */
  bool exist(const String& full_xpath) const;

  /*!
   * \brief Method allowing checking if an option is present.
   *
   * The option address is formatted as follows:
   * xpath_before_index[ANY_INDEX]/xpath_after_index
   *
   * xpath_before_index must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by ANY_INDEX).
   *
   * xpath_after_index must be in the following format:
   * ddd/eee
   * - no "/" at the beginning or the end.
   *
   * The indices are XML indices and these indices start at 1. The ANY_INDEX index is a special index designating all indices.
   *
   * \param xpath_before_index The address before the index.
   * \param xpath_after_index The address after the index.
   * \return true if the address is found in the list.
   */
  bool existAnyIndex(const String& xpath_before_index, const String& xpath_after_index) const;

  /*!
   * \brief Method allowing checking if an option is present.
   *
   * The option address is formatted as follows:
   * full_xpath[ANY_INDEX]
   *
   * The address must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by ANY_INDEX).
   *
   * The indices are XML indices and these indices start at 1. The ANY_INDEX index is a special index designating all indices.
   *
   * \param full_xpath The address to search for.
   * \return true if the address is found in the list.
   */
  bool existAnyIndex(const String& full_xpath) const;

  /*!
   * \brief Method allowing retrieval of the index or indices of the option.
   *
   * The option address is formatted as follows:
   * xpath_before_index[GET_INDEX]/xpath_after_index
   *
   * xpath_before_index must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by GET_INDEX).
   *
   * xpath_after_index must be in the following format:
   * ddd/eee
   * - no "/" at the beginning or the end.
   *
   * The indices are XML indices and these indices start at 1. The GET_INDEX index is a special index designating the indices to be retrieved.
   *
   * \param xpath_before_index The address before the index.
   * \param xpath_after_index The address after the index.
   * \param indexes The array that will contain the set of found indices
   * (this array is not cleared before use).
   */
  void indexesInParam(const String& xpath_before_index, const String& xpath_after_index, UniqueArray<Integer>& indexes) const;

  /*!
   * \brief Method allowing retrieval of the index or indices of the option.
   *
   * The option address is formatted as follows:
   * xpath_before_index[GET_INDEX]
   *
   * xpath_before_index must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by GET_INDEX).
   *
   * If the parameter allow_elems_after_index is enabled, addresses of the form:
   * xpath_before_index[GET_INDEX]/aaa/bbb
   * will also be searched.
   *
   * The indices are XML indices and these indices start at 1. The GET_INDEX index is a special index designating the indices to be retrieved.
   *
   * \param xpath_before_index The address before the index.
   * \param indexes The array that will contain the set of found indices
   * \param allow_elems_after_index Should elements after the index be checked?
   * (this array is not cleared before use).
   */
  void indexesInParam(const String& xpath_before_index, UniqueArray<Integer>& indexes, bool allow_elems_after_index) const;

  /*!
   * \brief Method allowing knowing the number of indices of the option.
   *
   * The option address is formatted as follows:
   * xpath_before_index[GET_INDEX]/xpath_after_index
   *
   * xpath_before_index must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by GET_INDEX).
   *
   * xpath_after_index must be in the following format:
   * ddd/eee
   * - no "/" at the beginning or the end.
   *
   * The indices are XML indices and these indices start at 1. The GET_INDEX index is a special index designating the indices to be retrieved.
   *
   * \param xpath_before_index The address before the index.
   * \param xpath_after_index The address after the index.
   * \return The number of indices of the option.
   */
  Integer count(const String& xpath_before_index, const String& xpath_after_index) const;

  /*!
   * \brief Method allowing knowing the number of indices of the option.
   *
   * The option address is formatted as follows:
   * xpath_before_index[GET_INDEX]
   *
   * xpath_before_index must be in the following format:
   * //case/aaa/bbb[2]/ccc
   * - the "//case/" at the beginning (or "//cas/" in French),
   * - a succession of tags possibly with their indices,
   * - no "/" at the end,
   * - an index may be placed at the end (but it will be replaced
   *   by GET_INDEX).
   *
   * The indices are XML indices and these indices start at 1. The GET_INDEX index is a special index designating the indices to be retrieved.
   *
   * \param xpath_before_index The address before the index.
   * \return The number of indices of the option.
   */
  Integer count(const String& xpath_before_index) const;

 private:

  inline StringView _removeUselessPartInXpath(StringView xpath) const;

 private:

  bool m_is_fr = false;
  ParameterOptionElementsCollection* m_lines = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
