// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplace.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Class allowing symbols in a character string to be replaced by another    */
/* character string defined in the launch arguments.                         */
/* A symbol is defined by a character string surrounded by @.                */
/* Example: @mon_symbole@                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/StringVariableReplace.h"

#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/String.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/List.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String StringVariableReplace::
replaceWithCmdLineArgs(StringView string_with_symbols, bool fatal_if_not_found, bool fatal_if_invalid)
{
  StringList args;
  platform::fillCommandLineArguments(args);
  const CommandLineArguments cla{ args };
  ParameterListWithCaseOption parameters;
  parameters.addParameters(cla.parameters());
  return replaceWithCmdLineArgs(parameters, string_with_symbols, fatal_if_not_found, fatal_if_invalid);
}

/*!
 * \brief Method allowing symbols in the character string \a string_with_symbols
 * to be replaced by their values defined in the parameter list.
 *
 * A symbol is represented by a character string surrounded by two "\@".
 *
 * Example: "\@mesh_dir\@/cube.msh"
 * with a parameter "mesh_dir=~/mesh"
 * results in: "~/mesh/cube.msh".
 *
 *
 * To prevent a "\@" from being replaced, it is possible to put a backslash before it.
 * The backslash will be removed by this method.
 *
 * Example: "\@mesh_dir\@/cube\\\@.msh"
 * with a parameter "mesh_dir=~/mesh"
 * results in: "~/mesh/cube\@.msh".
 *
 *
 * If the number of at signs is incorrect (excluding escaped at signs),
 * an error will be triggered, unless the parameter \a fatal_if_invalid is set to \a false.
 * In this case, the last at sign will simply be removed.
 *
 * Example: "\@mesh_dir\@\@/cube.msh"
 * with a parameter "mesh_dir=~/mesh"
 * results in: "~/mesh/cube.msh".
 *
 *
 * Symbols that are not found will be removed or, if the parameter
 * \a fatal_if_not_found is set to \a true, an error will be triggered.
 *
 * Example: "\@mesh_dir\@/cube.msh"
 * without parameters
 * results in: "/cube.msh".
 *
 * Finally, having a parameter whose name contains an at sign will be invalid.
 * (However, the value may contain at signs).
 *
 * Invalid example: parameter "mesh\@_dir=~/mesh"
 * Valid example: parameter "mesh_dir=~/\@/mesh"
 *
 * \param parameter_list The list of parameters to consider.
 * \param string_with_symbols The character string with symbols to replace.
 * \param fatal_if_not_found If a symbol is not found in the parameter list, an error
 * will be triggered
 * if this parameter is true.
 * \param fatal_if_invalid If the character string is incorrect, an error will be triggered
 * if this parameter is true. Otherwise, the result is not guaranteed.
 * \return The character string with symbols replaced by their values.
 */
String StringVariableReplace::
replaceWithCmdLineArgs(const ParameterListWithCaseOption& parameter_list, StringView string_with_symbols,
                       bool fatal_if_not_found, bool fatal_if_invalid)
{
  // If the environment variable ARCANE_REPLACE_SYMBOLS_IN_DATASET is not
  // defined, we do not modify the character string.
  if (platform::getEnvironmentVariable("ARCANE_REPLACE_SYMBOLS_IN_DATASET").null() &&
      parameter_list.getParameterOrNull("ARCANE_REPLACE_SYMBOLS_IN_DATASET").null()) {
    return string_with_symbols;
  }

  if (string_with_symbols.empty())
    return string_with_symbols;

  Integer nb_at = 0;
  Integer nb_at_with_escape = 0;

  // At signs can be escaped. It is necessary to differentiate them to get
  // the size of the segments array and to check the validity of the character string.
  _countChar(string_with_symbols, '@', nb_at, nb_at_with_escape);
  if (nb_at == 0 && nb_at_with_escape == 0)
    return string_with_symbols;

  // If the number of at signs is odd, there is necessarily an inconsistency.
  if (fatal_if_invalid && nb_at % 2 == 1) {
    ARCANE_FATAL("Invalid nb of @");
  }

  // An at sign separates the string into two segments. So the number of segments will
  // be "nb_at + 1".
  // We add the escaped at signs. These at signs are considered a special symbol and
  // generate two segments (so "nb_at_with_escape * 2").
  const Integer size_array_w_splits = (nb_at + 1) + (nb_at_with_escape * 2);

  // In a valid case, the number of segments must be odd.
  // In the case where the user disables fatal_if_invalid and the number of segments is even,
  // the final segment should not be considered a symbol (see the loop below to understand).
  const Integer max_index = size_array_w_splits % 2 == 0 ? size_array_w_splits - 1 : size_array_w_splits;

  SmallArray<StringView> string_splited(size_array_w_splits);
  _splitString(string_with_symbols, string_splited, '@');

  StringBuilder combined{};

  // In the array, segments with an odd position are the elements between at signs
  // and thus symbols.
  for (Integer i = 0; i < max_index; ++i) {
    StringView part{ string_splited[i] };
    if (part.empty())
      continue;

    // Segments with an even position are not symbols.
    // We also handle the special case of escaped at signs.
    if (i % 2 == 0 || part.bytes()[0] == '@') {
      combined.append(part);
    }
    else {
      String reference_input = parameter_list.getParameterOrNull(part);
      if (reference_input.null()) {
        if (fatal_if_not_found) {
          ARCANE_FATAL("Symbol @{0}@ not found in the parameter list", part);
        }
      }
      else {
        combined.append(reference_input);
      }
    }
  }

  // In the case where the user disables fatal_if_invalid, the last segment
  // is at an odd position but is not a symbol, so we add it here.
  if (size_array_w_splits % 2 == 0) {
    combined.append(string_splited[string_splited.size() - 1]);
  }

  return combined.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Method allowing the string "str_view" to be split into multiple segments.
 * The splits will occur between the chars "c".
 * The segments will be added to the "str_view_array" array.
 * Segments with an odd position are symbols.
 * Segments with an even position are not symbols.
 * In the case where the number of segments is even, the last segment will
 * not be a symbol.
 * In the case where an at sign is escaped, it will be placed at an odd position. This special symbol will be considered.
 *
 * \param str_view [IN] The character string to split.
 * \param str_view_array [OUT] The array that will contain the segments.
 * \param c The character delimiting the segments.
 */
void StringVariableReplace::
_splitString(StringView str_view, ArrayView<StringView> str_view_array, char c)
{
  /*
    aa@aa      -> "aa", "aa"                       2 (fatal_if_invalid)
    @aa@aa@aa@ -> "", "aa", "aa", "aa", ""         5
    @aa@@aa@   -> "", "aa", "", "aa", ""           5
    @          -> "", ""                           2 (fatal_if_invalid)
    @aa@       -> "", "aa", ""                     3
    @aa@aa\@aa -> "", "aa", "aa", "@", "aa"        5
    @aa@\@@aa@ -> "", "aa", "", "@", "", "aa", ""  7
    @aa@@aa@   -> "", "aa", "", "aa", ""           5
    \@aa       -> "", "@", "aa"                    3
    @aa@aa@aa  -> "", "aa", "aa", "aa"             4 (fatal_if_invalid)
    @aa@aa     -> "", "aa", "aa"                   3
 */

  Span<const Byte> str_span = str_view.bytes();

  Int64 offset = 0;
  Int64 len = str_view.length();
  Integer index = 0;
  bool previous_backslash = false;

  for (Int64 i = 0; i < len; ++i) {
    // If we find an at sign.
    if (str_span[i] == c) {
      // If this at sign is preceded by a backslash.
      if (previous_backslash) {
        // We record the segment we were traversing, without the backslash.
        str_view_array[index++] = str_view.subView(offset, i - 1 - offset);
        // We add the at sign.
        str_view_array[index++] = str_view.subView(i, 1);

        offset = i + 1;
        previous_backslash = false;
      }
      else {
        // We record the segment we were traversing, without the at sign.
        str_view_array[index++] = str_view.subView(offset, i - offset);

        offset = i + 1;
      }
    }
    // If we find a backslash.
    else if (str_span[i] == '\\') {
      // And if we had already found a backslash, we do nothing.
      if (previous_backslash)
        previous_backslash = false;
      else
        previous_backslash = true;
    }
    else {
      previous_backslash = false;
    }
  }
  // We add the last segment.
  str_view_array[index] = str_view.subView(offset, len - offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Method allowing counting the number of separator characters
 * in a character string.
 *
 * \param str_view The character string.
 * \param c The character separating the segments.
 * \param count_c The total number of character \a c
 * \param count_c_with_escape The number of characters \a c preceded by a backslash.
 */
void StringVariableReplace::
_countChar(StringView str_view, char c, Integer& count_c, Integer& count_c_with_escape)
{
  count_c = 0;
  count_c_with_escape = 0;
  bool previous_backslash = false;

  for (const Byte byte : str_view.bytes()) {
    if (byte == c) {
      if (previous_backslash) {
        count_c_with_escape++;
        previous_backslash = false;
      }
      else {
        count_c++;
      }
    }
    else if (byte == '\\') {
      // If we had already found a backslash, we do nothing.
      if (previous_backslash)
        previous_backslash = false;
      else
        previous_backslash = true;
    }
    else {
      previous_backslash = false;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
