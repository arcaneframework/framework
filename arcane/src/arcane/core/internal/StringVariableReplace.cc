// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplace.cc                                    (C) 2000-2023 */
/*                                                                           */
/* Classe permettant de remplacer les symboles d'une chaine de caractères    */
/* par une autre chaine de caractères définie dans les arguments de          */
/* lancement.                                                                */
/* Un symbole est défini par une chaine de caractères entourée de @.         */
/* Exemple : @mon_symbole@                                                   */
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/StringVariableReplace.h"
#include "arcane/utils/PlatformUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Méthode permettant de remplacer les symboles de la chaine de caractères name
 * par leurs valeurs définies dans la liste des paramètres.
 *
 * Pour l'instant, un symbole est représenté par une chaine de caractères entourée
 * de deux "\@".
 *
 * Exemple : "\@mesh_dir\@/cube.msh"
 * avec un paramètre "mesh_dir=~/mesh"
 * donnera : "~/mesh/cube.msh".
 *
 * À noter que les @ ne seront pas supprimés s'ils ne correspondent pas à un symbole.
 *
 * Exemple : "\@mesh_dir\@/cube.msh"
 * sans paramètres
 * donnera : "\@mesh_dir\@/cube.msh".
 *
 * @param parameter_list La liste des paramètres à considérer.
 * @param string_with_symbols La chaine de caractères avec les symboles à remplacer.
 * @return La chaine de caractères avec les symboles remplacés par leurs valeurs.
 */
String StringVariableReplace::
replaceWithCmdLineArgs(const ParameterList& parameter_list, const String& string_with_symbols)
{
  // Si la variable d'environnement ARCANE_REPLACE_SYMBOLS_IN_DATASET n'est pas
  // définie, on ne touche pas à la chaine de caractères.
  if (platform::getEnvironmentVariable("ARCANE_REPLACE_SYMBOLS_IN_DATASET").null() &&
      parameter_list.getParameterOrNull("ARCANE_REPLACE_SYMBOLS_IN_DATASET").null()) {
    return string_with_symbols;
  }

  if(string_with_symbols.empty()) return string_with_symbols;

  UniqueArray<String> string_splited;
  UniqueArray<Integer> symbol_pos;
  Integer nb_at;

  // On découpe la string là où se trouve les @.
  // On récupère la position des symboles et le nombre de @.
  _splitString(string_with_symbols, string_splited, symbol_pos, nb_at, '@');

  // S'il n'y a aucun symbole, on retourne la chaine d'origine.
  if(symbol_pos.empty()) return string_with_symbols;

  for(Integer pos : symbol_pos){
    String& symbol = string_splited[pos];
    String reference_input = parameter_list.getParameterOrNull(symbol);
    if(reference_input.null()) {
      symbol = "@" + symbol + "@";
    }
    else {
      symbol = reference_input;
    }
  }

  // On recombine la chaine de caractères.
  StringBuilder combined = "";
  for(Integer i = 0; i < string_splited.size() - 1; ++i){
    const String& str = string_splited[i];
    combined.append(str);
  }

  // Si le nombre de @ est impair, alors il faut rajouter un @ avant le dernier element pour reproduire
  // la chaine d'origine.
  if(nb_at % 2 != 0){
    combined.append("@" + string_splited[string_splited.size()-1]);
  }
  else{
    combined.append(string_splited[string_splited.size()-1]);
  }

  return combined.toString();
}

/*!
 * Méthode permettant de splitter la chaine "str" en plusieurs morceaux.
 * Les splits seront entre les chars "c".
 * Les morceaux seront ajoutés dans le tableau "str_array".
 * Les positions des morceaux correspondant à un symbole seront ajoutées
 * dans le tableau "int_array".
 * Le nombre de char "c" sera mis dans le paramètre nb_c.
 *
 * @param str [IN] La chaine de caractères à split.
 * @param str_array [OUT] Le tableau qui contiendra les morceaux.
 * @param int_array [OUT] Le tableau avec les positions des symboles dans le tableau "str_array".
 * @param nb_c [OUT] Le nombre de char "c".
 * @param c Le char délimitant les morceaux.
 */
void StringVariableReplace::_splitString(const String& str, UniqueArray<String>& str_array, UniqueArray<Integer>& int_array, Integer& nb_c, char c)
{
  Span<const Byte> str_str = str.bytes();

  Int64 offset = 0;
  bool is_symbol = false;
  Int64 len = str.length();
  nb_c = 0;

  for(Int64 i = 0; i < len; ++i) {

    if (str_str[i] == c) {
      nb_c++;
      str_array.add(str.substring(offset, i - offset));
      offset = i+1;

      if(is_symbol){
        int_array.add(str_array.size() - 1);
        is_symbol = false;
      }
      else{
        is_symbol = true;
      }
    }
  }
  if(offset == len){
    str_array.push_back("");
  }
  else {
    str_array.push_back(str.substring(offset, len - offset));
  }
}


//template<typename StringContainer>
//void StringVariableReplace::split(const String& str, StringContainer& str_array, char c)
//{
//  Span<const Byte> str_str = str.bytes();
//
//  Int64 offset = 0;
//  Int64 len = str.length();
//
//
//  for( Int64 i = 0; i < len; ++i ) {
//    if (str_str[i] == c) {
//      str_array.push_back(str.substring(offset, i - offset));
//      offset = i+1;
//
//    }
//  }
//  str_array.push_back(str.substring(offset, len - offset));
//}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
