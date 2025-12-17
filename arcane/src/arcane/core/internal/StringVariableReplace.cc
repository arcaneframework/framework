// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplace.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de remplacer les symboles d'une chaine de caractères    */
/* par une autre chaine de caractères définie dans les arguments de          */
/* lancement.                                                                */
/* Un symbole est défini par une chaine de caractères entourée de @.         */
/* Exemple : @mon_symbole@                                                   */
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
 * \brief Méthode permettant de remplacer les symboles de la chaine de
 * caractères \a string_with_symbols par leurs valeurs définies dans la liste
 * des paramètres.
 *
 * Un symbole est représenté par une chaine de caractères entourée
 * de deux "\@".
 *
 * Exemple : "\@mesh_dir\@/cube.msh"
 * avec un paramètre "mesh_dir=~/mesh"
 * donnera : "~/mesh/cube.msh".
 *
 *
 * Pour éviter qu'un "\@" soit remplacé, il est possible de mettre un backslash avant.
 * Le backslash sera supprimé par cette méthode.
 *
 * Exemple : "\@mesh_dir\@/cube\\\@.msh"
 * avec un paramètre "mesh_dir=~/mesh"
 * donnera : "~/mesh/cube\@.msh".
 *
 *
 * Si le nombre d'arrobases est incorrect (sans compter les arrobases échappées),
 * une erreur sera déclenchée, sauf si le paramètre \a fatal_if_invalid est à \a false.
 * Dans ce cas, la dernière arrobase sera simplement supprimée.
 *
 * Exemple : "\@mesh_dir\@\@/cube.msh"
 * avec un paramètre "mesh_dir=~/mesh"
 * donnera : "~/mesh/cube.msh".
 *
 *
 * Les symboles qui ne sont pas trouvés seront supprimés ou, si le paramètre
 * \a fatal_if_not_found est à \a true, une erreur sera déclenchée.
 *
 * Exemple : "\@mesh_dir\@/cube.msh"
 * sans paramètres
 * donnera : "/cube.msh".
 *
 * Enfin, avoir un paramètre ayant un nom contenant une arrobase sera invalide.
 * (en revanche, la valeur peut contenir des arrobases).
 *
 * Exemple invalide : paramètre "mesh\@_dir=~/mesh"
 * Exemple valide : paramètre "mesh_dir=~/\@/mesh"
 *
 * \param parameter_list La liste des paramètres à considérer.
 * \param string_with_symbols La chaine de caractères avec les symboles à remplacer.
 * \param fatal_if_not_found Si un symbole n'est pas trouvé dans la liste des paramètres, une erreur sera déclenchée
 * si ce paramètre vaut true.
 * \param fatal_if_invalid Si la chaine de caractères est incorrecte, une erreur sera déclenchée
 * si ce paramètre vaut true. Sinon, le résultat n'est pas garanti.
 * \return La chaine de caractères avec les symboles remplacés par leurs valeurs.
 */
String StringVariableReplace::
replaceWithCmdLineArgs(const ParameterListWithCaseOption& parameter_list, StringView string_with_symbols,
                       bool fatal_if_not_found, bool fatal_if_invalid)
{
  // Si la variable d'environnement ARCANE_REPLACE_SYMBOLS_IN_DATASET n'est pas
  // définie, on ne touche pas à la chaine de caractères.
  if (platform::getEnvironmentVariable("ARCANE_REPLACE_SYMBOLS_IN_DATASET").null() &&
      parameter_list.getParameterOrNull("ARCANE_REPLACE_SYMBOLS_IN_DATASET").null()) {
    return string_with_symbols;
  }

  if (string_with_symbols.empty())
    return string_with_symbols;

  Integer nb_at = 0;
  Integer nb_at_with_escape = 0;

  // Les arrobases peuvent être échappées. Il est nécessaire de les différencier pour avoir
  // la taille du tableau des morceaux et pour vérifier la validité de la chaine de caractères.
  _countChar(string_with_symbols, '@', nb_at, nb_at_with_escape);
  if (nb_at == 0 && nb_at_with_escape == 0)
    return string_with_symbols;

  // Si le nombre d'arrobases est impaire, il y a forcément une incohérence.
  if (fatal_if_invalid && nb_at % 2 == 1) {
    ARCANE_FATAL("Invalid nb of @");
  }

  // Une arrobase sépare la chaine en deux morceaux. Donc le nombre de morceaux sera de "nb_at + 1".
  // On ajoute les arrobases échappées. Ces arrobases sont considérées comme un symbole spécial et
  // génèrent deux morceaux (donc "nb_at_with_escape * 2").
  const Integer size_array_w_splits = (nb_at + 1) + (nb_at_with_escape * 2);

  // Dans un cas valide, le nombre de morceaux doit être impaire.
  // Dans le cas où l'utilisateur désactive le fatal_if_invalid et que le nombre de morceaux est paire,
  // il ne faut pas qu'on considère le morceau final comme un symbole (voir la boucle plus bas pour comprendre).
  const Integer max_index = size_array_w_splits % 2 == 0 ? size_array_w_splits - 1 : size_array_w_splits;

  SmallArray<StringView> string_splited(size_array_w_splits);
  _splitString(string_with_symbols, string_splited, '@');

  StringBuilder combined{};

  // Dans le tableau, les morceaux ayant une position impaire sont les élements entre arrobases
  // et donc des symboles.
  for (Integer i = 0; i < max_index; ++i) {
    StringView part{ string_splited[i] };
    if (part.empty())
      continue;

    // Les morceaux avec une position paire ne sont pas des symboles.
    // On traite aussi le cas particulier des arrobases échappées.
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

  // Dans le cas où l'utilisateur désactive le fatal_if_invalid, le dernier morceau
  // est à une position impaire mais n'est pas un symbole, donc on le rajoute ici.
  if (size_array_w_splits % 2 == 0) {
    combined.append(string_splited[string_splited.size() - 1]);
  }

  return combined.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Méthode permettant de splitter la chaine "str_view" en plusieurs morceaux.
 * Les splits seront entre les chars "c".
 * Les morceaux seront ajoutés dans le tableau "str_view_array".
 * Les morceaux ayant une position impaire sont des symboles.
 * Les morceaux ayant une position paire ne sont pas des symboles.
 * Dans le cas où le nombre de morceaux est paire, le dernier morceau ne sera
 * pas un symbole.
 * Dans le cas où une arrobase est échappée, elle sera mise à une position
 * impaire. Ce symbole spécial sera à considérer.
 *
 * \param str_view [IN] La chaine de caractères à split.
 * \param str_view_array [OUT] Le tableau qui contiendra les morceaux.
 * \param c Le char délimitant les morceaux.
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
    // Si on trouve une arrobase.
    if (str_span[i] == c) {
      // Si cette arrobase est précédée par un backslash.
      if (previous_backslash) {
        // On enregistre le morceau qu'on parcourait, sans le backslash.
        str_view_array[index++] = str_view.subView(offset, i - 1 - offset);
        // On rajoute l'arrobase.
        str_view_array[index++] = str_view.subView(i, 1);

        offset = i + 1;
        previous_backslash = false;
      }
      else {
        // On enregistre le morceau qu'on parcourait, sans l'arrobase.
        str_view_array[index++] = str_view.subView(offset, i - offset);

        offset = i + 1;
      }
    }
    // Si on trouve un backslash.
    else if (str_span[i] == '\\') {
      // Et qu'on avait déjà trouvé un backslash, on n'y touche pas.
      if (previous_backslash)
        previous_backslash = false;
      else
        previous_backslash = true;
    }
    else {
      previous_backslash = false;
    }
  }
  // On ajoute le dernier morceau.
  str_view_array[index] = str_view.subView(offset, len - offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Méthode permettant de compter le nombre de caractères séparateurs
 * dans une chaine de caractères.
 *
 * \param str_view La chaine de caractère.
 * \param c Le caractère séparant les morceaux.
 * \param count_c Le nombre total de caractère \a c
 * \param count_c_with_escape Le nombre de caractères \a c ayant un backslash avant.
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
      // On avait déjà trouvé un backslash, on n'y touche pas.
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
