// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterOption.h                                           (C) 2000-2025 */
/*                                                                           */
/* Class representing the set of parameters that can modify the              */
/* data set options.                                                         */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_UTILS_INTERNAL_PARAMETEROPTION_H
#define ARCANE_UTILS_INTERNAL_PARAMETEROPTION_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class representing a part of a data set option address.
 * Note that in XML, the index starts at 1 and not at 0.
 *
 * A special tag named ANY_TAG represents any tag.
 * Two special indices are also available:
 * - ANY_INDEX: Represents any index,
 * - GET_INDEX: Represents an index to be retrieved (see the ParameterOptionAddr class).
 * These elements are useful for the == operator.
 * Note that ANY_TAG cannot be defined without ANY_INDEX.
 * Also, the tag cannot be empty.
 */
class ARCANE_UTILS_EXPORT
ParameterOptionAddrPart
{
 public:
  static constexpr const char* ANY_TAG = "/";
  static constexpr Integer ANY_INDEX = -1;
  static constexpr Integer GET_INDEX = -2;

 public:

  /*!
   * \brief Constructor. Sets the tag to ANY_TAG and the index to ANY_INDEX.
   */
  ParameterOptionAddrPart();

  /*!
   * \brief Constructor. Sets the index to 1.
   * \param tag The tag of this address part. This tag cannot be ANY_TAG.
   */
  explicit ParameterOptionAddrPart(const StringView tag);

  /*!
   * \brief Constructor.
   * \param tag The tag of this address part. This tag cannot be ANY_TAG
   * if the index is not ANY_INDEX.
   * \param index The index of this address part.
   */
  ParameterOptionAddrPart(const StringView tag, const Integer index);

 public:

  StringView tag() const;
  Integer index() const;

  //! If the index is ANY_INDEX, the tag cannot be ANY_TAG.
  //! Be careful about the lifetime of tag.
  void setTag(const StringView tag);
  void setIndex(const Integer index);

  //! isAny if ANY_TAG and ANY_INDEX.
  bool isAny() const;

  /*!
   * \brief Equality operator.
   * The ANY_TAG tag is equal to all tags.
   * The ANY_INDEX index is equal to all indices.
   * The GET_INDEX index is equal to all indices.
   */
  bool operator==(const ParameterOptionAddrPart& other) const;
  // TODO AH: To be removed when migrating to C++20.
  bool operator!=(const ParameterOptionAddrPart& other) const;

 private:

  StringView m_tag;
  Integer m_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class representing a data set option address.
 * This address must be in the form: "tag/tag[index]/tag"
 * Address parts without an index will have the default index (=1).
 *
 * This address must comply with certain rules:
 * - it must not be empty,
 * - it must not represent all options ("/"),
 * - its tags can be empty if the index is empty (see below),
 * - the special index ANY_INDEX can only be present if the tag is not empty,
 * - the address can end with an attribute ("\@name"),
 * - the address given to the constructor cannot end with an ANY_TAG (but
 *   ANY_TAG can be added later with the addAddrPart() method),
 *
 * In a character string:
 * - the pattern ANY_TAG[ANY_INDEX] can be defined with "//":
 *   -> "tag/tag//tag" will be converted as: "tag[1]/tag[1]/ANY_TAG[ANY_INDEX]/tag[1]".
 * - the ANY_INDEX can be defined with an empty index "[]":
 *   -> "tag/tag[]/\@attr" will be converted as: "tag[1]/tag[ANY_INDEX]/\@attr[1]",
 *   -> the pattern "tag/[]/tag" is forbidden.
 */
class ARCANE_UTILS_EXPORT
ParameterOptionAddr
{
 public:

  /*!
   * \brief Constructor.
   * \param addr_str_view The address to convert.
   */
  explicit ParameterOptionAddr(StringView addr_str_view);

 public:

  // We must not block multiple ParameterOptionAddrPart(ANY):
  // Construction by iteration: aaaa/bb/ANY/ANY/cc
  /*!
   * \brief Method allowing a part to be added to the end of the current address.
   * \param part A pointer to the new part. Note that we take ownership of the
   * object (we manage the delete).
   */
  void addAddrPart(ParameterOptionAddrPart* part);

  /*!
   * \brief Method allowing a part of the address to be retrieved.
   * If the address ends with an ANY_TAG[ANY_INDEX], all indices given in the parameter
   * greater than the number of parts of the address will return the last element of
   * the address ("ANY_TAG[ANY_INDEX]").
   *
   * \param index_of_part The index of the part to retrieve.
   * \return The part of the address.
   */
  ParameterOptionAddrPart* addrPart(const Integer index_of_part) const;

  ParameterOptionAddrPart* lastAddrPart() const;

  /*!
   * \brief Method allowing the number of parts of the address to be retrieved.
   * Parts equal to "ANY_TAG[ANY_INDEX]" are counted.
   *
   * \return The number of parts of the address.
   */
  Integer nbAddrPart() const;

  /*!
   * \brief Method allowing one or more indices to be retrieved in the address.
   *
   * The functioning of this method is simple.
   * We have the following address: "aaa[1]/bbb[2]/ccc[4]/@name[1]".
   * The address in the parameter is the following: "aaa[1]/bbb[GET_INDEX]/ccc[4]/@name[1]".
   * The index added in the parameter view will be 2.
   *
   * If the address in the parameter is: "aaa[1]/bbb[GET_INDEX]/ccc[GET_INDEX]/@name[1]".
   * The indices added in the view will be 2 and 4.
   *
   * Conversely, a "GET_INDEX" cannot be used on an "ANY_INDEX" (returns false).
   * Example: if we have: "aaa[1]/bbb[ANY_INDEX]/ccc[4]/@name[1]".
   * And if the address in the parameter is: "aaa[1]/bbb[GET_INDEX]/ccc[GET_INDEX]/@name[1]".
   * The returned boolean will be false.
   *
   * To get the correct size of the view, a call to the method "nbIndexToGetInAddr()"
   * can be made.
   *
   * \param addr_with_get_index The address containing "GET_INDEX" indices.
   * \param indexes [OUT] The view where the index or indices will be added (the size must be correct).
   * \return true if the view could be filled correctly.
   */
  bool getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, ArrayView<Integer> indexes) const;

  /*!
   * \brief Method allowing the number of "GET_INDEX" in the address to be known.
   * \return The number of "GET_INDEX".
   */
  Integer nbIndexToGetInAddr() const;

 public:

  /*!
   * \brief Equality operator.
   * This operator takes into account ANY_TAG / ANY_INDEX.
   * The address "aaa[1]/bbb[2]/ANY_TAG[ANY_INDEX]"
   * will be equal to the address "aaa[1]/bbb[2]/ccc[5]/ddd[7]"
   * or to the address "aaa[1]/bbb[ANY_INDEX]/ccc[5]/ddd[7]"
   * or to the address "aaa[1]/bbb[2]"
   * but not to the address "aaa[1]"
   */
  bool operator==(const ParameterOptionAddr& other) const;

  // TODO AH: To be removed when migrating to C++20.
  bool operator!=(const ParameterOptionAddr& other) const;

 private:

  UniqueArray<Ref<ParameterOptionAddrPart>> m_parts;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class representing an XML element (an Arcane option).
 * This element has an address and a value.
 */
class ARCANE_UTILS_EXPORT
ParameterOptionElement
{
 public:

  ParameterOptionElement(const StringView addr, const StringView value);

  ParameterOptionAddr addr() const;

  StringView value() const;

  bool operator==(const ParameterOptionAddr& addr) const;

 private:

  ParameterOptionAddr m_addr;
  StringView m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class representing a collection of XML elements (a set of Arcane options).
 */
class ARCANE_UTILS_EXPORT
ParameterOptionElementsCollection
{
 public:

  /*!
   * \brief Method allowing an option parameter to be added to the list
   * of option parameters.
   *
   * \warning The two parameters are not copied! Only a view is retrieved. The user
   * of this class must manage the lifetime of these objects.
   *
   * \param parameter The raw option parameter (with "//" at the beginning).
   * \param value The value of the option.
   */
  void addParameter(const String& parameter, const String& value);

  void addElement(StringView addr, StringView value);

  // ParameterOptionElement element(const Integer index)
  // {
  //   return m_elements[index];
  // }

  // An empty StringView is equal to a null StringView.
  // Since we are working with Strings and the distinction
  // empty/null is important, we use std::optional.
  std::optional<StringView> value(const ParameterOptionAddr& addr);

  /*!
   * \brief Method allowing to know if an address is present in the list of elements.
   * ANY_TAG/ANY_INDEX are taken into account.
   * \param addr The address to search for.
   * \return true if the address is found.
   */
  bool isExistAddr(const ParameterOptionAddr& addr);

  /*!
   * \brief Method allowing to know how many times an address is present in the list of elements.
   * Method particularly useful with ANY_TAG/ANY_INDEX.
   *
   * \param addr The address to search for.
   * \return The number of matches found.
   */
  Integer countAddr(const ParameterOptionAddr& addr);

  /*!
   * \brief Method allowing one or more indices to be retrieved in the list of addresses.
   *
   * The functioning of this method is simple.
   * We have the following addresses: "aaa[1]/bbb[2]/ccc[1]/@name[1]".
   *                                          "aaa[1]/bbb[2]/ccc[2]/@name[1]".
   *                                          "ddd[1]/eee[2]".
   *                                          "fff[1]/ggg[2]/hhh[4]".
   * The address in the parameter is the following: "aaa[1]/bbb[2]/ccc[GET_INDEX]/@name[1]".
   * The indices added in the parameter array will be 1 and 2.
   *
   * Warning: Having an input address with multiple "GET_INDEX" is allowed but
   * it can be dangerous if the number of indices found per address is different for
   * each address (if there are two "GET_INDEX" but in one of the addresses, there are not
   * two matches, these potential matches will not be taken into
   * account).
   *
   * \param addr_with_get_index The address containing "GET_INDEX" indices.
   * \param indexes [OUT] The array where the index or indices will be added (the array
   * is not cleared before use).
   */
  void getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, UniqueArray<Integer>& indexes);

 private:

  UniqueArray<ParameterOptionElement> m_elements;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
