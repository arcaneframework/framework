// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableOutput.hh                                       (C) 2000-2022 */
/*                                                                           */
/* Interface pour simples services de sortie de tableaux de valeurs.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEOUTPUT_H
#define ARCANE_ISIMPLETABLEOUTPUT_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISimpleTableInternalMng.h"
#include "arcane/core/ISimpleTableWriterHelper.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @ingroup StandardService
 * @brief Interface représentant une sortie de tableau simple.
 */
class ARCANE_CORE_EXPORT ISimpleTableOutput
{
 public:
  virtual ~ISimpleTableOutput() = default;

 public:
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   */
  virtual bool init() = 0;
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   * 
   * @param table_name Le nom du tableau (et du fichier de sortie).
   */
  virtual bool init(const String& table_name) = 0;
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   * 
   * @param table_name Le nom du tableau (et du fichier de sortie).
   * @param directory_name Le nom du dossier dans lequel enregistrer les tableaux.
   */
  virtual bool init(const String& table_name, const String& directory_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de remettre à zéro les tableaux
   */
  virtual void clear() = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter une ligne.
   * 
   * @param row_name Le nom de la ligne.
   * @return Integer La position de la ligne dans le tableau.
   */
  virtual Integer addRow(const String& row_name) = 0;
  /**
   * @brief Méthode permettant d'ajouter une ligne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés).
   * 
   * @param row_name Le nom de la ligne.
   * @param elements Les éléments à insérer sur la ligne.
   * @return Integer La position de la ligne dans le tableau.
   */
  virtual Integer addRow(const String& row_name, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs lignes.
   * 
   * @param rows_names Les noms des lignes.
   * @return true Si toutes les lignes ont été créées.
   * @return false Si toutes les lignes n'ont pas été créées.
   */
  virtual bool addRows(StringConstArrayView rows_names) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter une colonne.
   * 
   * @param column_name Le nom de la colonne.
   * @return Integer La position de la colonne dans le tableau.
   */
  virtual Integer addColumn(const String& column_name) = 0;
  /**
   * @brief Méthode permettant d'ajouter une colonne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés).
   * 
   * @param column_name Le nom de la colonne.
   * @param elements Les éléments à ajouter sur la colonne.
   * @return Integer La position de la colonne dans le tableau.
   */
  virtual Integer addColumn(const String& column_name, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs colonnes.
   * 
   * @param rows_names Les noms des colonnes.
   * @return true Si toutes les colonnes ont été créées.
   * @return false Si toutes les colonnes n'ont pas été créées.
   */
  virtual bool addColumns(StringConstArrayView columns_names) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter un élément à une ligne.
   * 
   * @param position La position de la ligne.
   * @param element L'élément à ajouter.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInRow(Integer position, Real element) = 0;
  /**
   * @brief Méthode permettant l'ajouter un élément sur une ligne.
   * 
   * @param row_name Le nom de la ligne.
   * @param element L'élément à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la 
   *                            ligne si elle n'existe pas encore.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInRow(const String& row_name, Real element, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter un élément sur la ligne 
   * dernièrement manipulée.
   * 
   * Cette méthode diffère de 'editElementRight()' car ici, on ajoute 
   * un élément à la fin de la ligne, pas forcement après le
   * dernier élement ajouté.
   * 
   * @param element L'élément à ajouter.
   * @return true Si l'élément a été ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInSameRow(Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une ligne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param position La position de la ligne.
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInRow(Integer position, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une ligne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param row_name Le nom de la ligne.
   * @param elements Le tableau d'élement à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la ligne
   *                            si elle n'existe pas encore.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInRow(const String& row_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur la 
   * ligne dernièrement manipulée.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * Mis à part le fait qu'ici, on manipule un tableau, cette méthode diffère
   * de 'editElementRight()' car ici, on ajoute des éléments à la fin de la ligne,
   * pas forcement après le dernier élement ajouté.
   * 
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInSameRow(ConstArrayView<Real> elements) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter un élément à une colonne.
   * 
   * @param position La position de la colonne.
   * @param element L'élément à ajouter.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInColumn(Integer position, Real element) = 0;
  /**
   * @brief Méthode permettant l'ajouter un élément sur une colonne.
   * 
   * @param column_name Le nom de la colonne.
   * @param element L'élément à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la colonne
   *                            si elle n'existe pas encore.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInColumn(const String& column_name, Real element, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter un élément sur la colonne
   * dernièrement manipulée.
   * 
   * Cette méthode diffère de 'editElementDown()' car ici, on ajoute un élément
   * à la fin de la colonne, pas forcement après le dernier élement ajouté.
   * 
   * @param element L'élément à ajouter.
   * @return true Si l'élément a été ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInSameColumn(Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une colonne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param position La position de la colonne.
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInColumn(Integer position, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une colonne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param column_name Le nom de la colonne.
   * @param elements Le tableau d'élement à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la colonne si
   *                            elle n'existe pas encore.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInColumn(const String& column_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur la
   * colonne dernièrement manipulée.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * Mis à part le fait qu'ici, on manipule un tableau, cette méthode diffère
   * de 'editElementDown()' car ici, on ajoute des éléments à la fin de la colonne,
   * pas forcement après le dernier élement ajouté.
   * 
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInSameColumn(ConstArrayView<Real> elements) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'éditer un élément au-dessus du dernier
   * élement dernièrement manipulé (ligne du dessus/même colonne).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementUp(Real element, bool update_last_position = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément en-dessous du dernier 
   * élement dernièrement manipulé (ligne du dessous/même colonne).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié 
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * Cette méthode diffère de 'addElementInSameColumn()' car ici, on ajoute 
   * (ou modifie) un élement sous le dernier élement manipulé, qui n'est
   * pas forcement à la fin de la colonne.
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementDown(Real element, bool update_last_position = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément à gauche du dernier
   * élement dernièrement manipulé (même ligne/colonne à gauche).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementLeft(Real element, bool update_last_position = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément à droite du dernier
   * élement dernièrement manipulé (même ligne/colonne à droite).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * Cette méthode diffère de 'addElementInSameRow()' car ici, on ajoute 
   * (ou modifie) un élement à la droite du dernier élement manipulé, 
   * qui n'est pas forcement à la fin de la colonne.
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementRight(Real element, bool update_last_position = true) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer un élément au-dessus du dernier
   * élement dernièrement manipulé (ligne du dessus/même colonne).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementUp(bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément en-dessous du dernier
   * élement dernièrement manipulé (ligne du dessous/même colonne).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementDown(bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément à gauche du dernier
   * élement dernièrement manipulé (même ligne/colonne à gauche).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementLeft(bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément à droite du dernier
   * élement dernièrement manipulé (même ligne/colonne à droite).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementRight(bool update_last_position = false) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * Les positions x et y correspondent à l'emplacement du dernier
   * élement manipulé.
   * 
   * Cette méthode a de l'intéret après l'utilisation de
   * 'elemUDLR(true)' par exemple.
   * 
   * @param element L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas été remplacé.
   */
  virtual bool editElement(Real element) = 0;
  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * @param position_x La position de la colonne à modifier.
   * @param position_y La position de la ligne à modifier.
   * @param element L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas été remplacé.
   */
  virtual bool editElement(Integer position_x, Integer position_y, Real element) = 0;
  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * @param column_name Le nom de la colonne où se trouve l'élement.
   * @param row_name Le nom de la ligne où se trouve l'élement.
   * @param element L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas pu être remplacé.
   */
  virtual bool editElement(const String& column_name, const String& row_name, Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * Les positions x et y correspondent à l'emplacement du dernier élement manipulé.
   * 
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real element() = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * @param position_x La position de la colonne où se trouve l'élement.
   * @param position_y La position de la ligne où se trouve l'élement.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real element(Integer position_x, Integer position_y, bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * @param column_name Le nom de la colonne où se trouve l'élement.
   * @param row_name Le nom de la ligne où se trouve l'élement.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real element(const String& column_name, const String& row_name, bool update_last_position = false) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir une copie d'une ligne.
   * 
   * @param position La position de la ligne.
   * @return RealUniqueArray La copie de la ligne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray row(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'une ligne.
   * 
   * @param row_name Le nom de la ligne.
   * @return RealUniqueArray La copie de la ligne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray row(const String& row_name) = 0;

  /**
   * @brief Méthode permettant d'obtenir une copie d'une colonne.
   * 
   * @param position La position de la colonne.
   * @return RealUniqueArray La copie de la colonne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray column(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'une colonne.
   * 
   * @param column_name Le nom de la colonne.
   * @return RealUniqueArray La copie de la colonne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray column(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir la taille d'une ligne.
   * Incluant les hypothétiques 'trous' dans la ligne.
   * 
   * @param position La position de la ligne.
   * @return Integer La taille de la ligne (0 si non trouvée).
   */
  virtual Integer rowSize(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir la taille d'une ligne.
   * Incluant les hypotétiques 'trous' dans la ligne.
   * 
   * @param position Le nom de la ligne.
   * @return Integer La taille de la ligne (0 si non trouvée).
   */
  virtual Integer rowSize(const String& row_name) = 0;

  /**
   * @brief Méthode permettant d'obtenir la taille d'une colonne.
   * Incluant les hypotétiques 'trous' dans la colonne.
   * 
   * @param position La position de la colonne.
   * @return Integer La taille de la colonne (0 si non trouvée).
   */
  virtual Integer columnSize(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir la taille d'une colonne.
   * Incluant les hypotétiques 'trous' dans la colonne.
   * 
   * @param position Le nom de la colonne.
   * @return Integer La taille de la colonne (0 si non trouvée).
   */
  virtual Integer columnSize(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer la position d'une ligne.
   * 
   * @param row_name Le nom de la ligne.
   * @return Integer La position de la ligne (-1 si non trouvée).
   */
  virtual Integer rowPosition(const String& row_name) = 0;
  /**
   * @brief Méthode permettant de récupérer la position d'une colonne.
   * 
   * @param row_name Le nom de la colonne.
   * @return Integer La position de la colonne (-1 si non trouvée).
   */
  virtual Integer columnPosition(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer le nombre de lignes dans le tableau.
   * C'est, en quelque sorte, le nombre max d'élements que peut contenir une colonne.
   * 
   * @return Integer Le nombre de lignes du tableau.
   */
  virtual Integer numberOfRows() = 0;
  /**
   * @brief Méthode permettant de récupérer le nombre de colonnes dans le tableau.
   * C'est, en quelque sorte, le nombre max d'élements que peut contenir une ligne.
   * 
   * @return Integer Le nombre de colonnes du tableau.
   */
  virtual Integer numberOfColumns() = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  virtual String rowName(Integer position) = 0;
  virtual String columnName(Integer position) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de changer le nom d'une ligne.
   * 
   * @param position La position de la ligne.
   * @param new_name Le nouveau nom de la ligne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editRowName(Integer position, const String& new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une ligne.
   * 
   * @param row_name Le nom actuel de la ligne.
   * @param new_name Le nouveau nom de la ligne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editRowName(const String& row_name, const String& new_name) = 0;

  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param position La position de la colonne.
   * @param new_name Le nouveau nom de la colonne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editColumnName(Integer position, const String& new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param column_name Le nom actuel de la colonne.
   * @param new_name Le nouveau nom de la colonne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editColumnName(const String& column_name, const String& new_name) = 0;

  /**
   * @brief Méthode permettant de créer une colonne contenant la moyenne des
   * éléments de chaque ligne.
   * 
   * @param column_name Le nom de la nouvelle colonne.
   * @return Integer La position de la colonne.
   */
  virtual Integer addAverageColumn(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /**
   * @brief Méthode permettant d'afficher le tableau.
   * Méthode effectuant des opérations collectives.
   * 
   * @param rank L'id du processus devant afficher le tableau (-1 pour 
   *                  signifier "tous les processus").
   */
  virtual void print(Integer rank = 0) = 0;

  virtual bool writeFile(const Directory& root_directory, Integer rank) = 0;

  /**
   * @brief Méthode permettant d'écrire le tableau dans un fichier.
   * Méthode effectuant des opérations collectives.
   * Si rank != -1, les processus autres que P0 retournent true.
   * 
   * @param rank L'id du processus devant écrire dans un fichier 
   *                  le tableau (-1 pour signifier "tous les processus").
   * @return true Si le fichier a été correctement écrit.
   * @return false Si le fichier n'a pas été correctement écrit.
   */
  virtual bool writeFile(Integer rank = -1) = 0;
  /**
   * @brief Méthode permettant d'écrire le tableau dans un fichier.
   * Méthode effectuant des opérations collectives.
   * Si rank != -1, les processus autres que P0 retournent true.
   * 
   * @param directory Le répertoire où sera écrit le fichier
   *            . Le chemin final sera "./[output_dir]/csv/[directory]/"
   * @param rank L'id du processus devant écrire dans un fichier 
   *                  le tableau (-1 pour signifier "tous les processus").
   * @return true Si le fichier a été correctement écrit.
   * @return false Si le fichier n'a pas été correctement écrit.
   * 
   * @deprecated Utiliser setOutputDirectory() puis writeFile() à la place.
   */
  virtual bool writeFile(const String& directory, Integer rank = -1) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer la précision actuellement
   * utilisée pour l'écriture des valeurs.
   * 
   * @return Integer La précision.
   */
  virtual Integer precision() = 0;
  /**
   * @brief Méthode permettant de modifier la précision du print.
   * 
   * Aussi bien pour la méthode 'print()' que les méthodes 'writeFile()'.
   * 
   * @warning Le flag "std::fixed" modifie le comportement de "setPrecision()",
   *          si le flag "std::fixed" est désactivé, la précision définira le
   *          nombre de chiffres total (avant et après la virgule) ;
   *          si le flag "std::fixed" est activé, la précision définira le
   *          nombre de chiffres après la virgule. Attention donc lors de
   *          l'utilisation de "std::numeric_limits<Real>::max_digits10"
   *          (pour l'écriture) ou de "std::numeric_limits<Real>::digits10"
   *          (pour la lecture) qui sont à utiliser sans le flag "std::fixed".
   * 
   * @param precision La nouvelle précision.
   */
  virtual void setPrecision(Integer precision) = 0;

  /**
   * @brief Méthode permettant de savoir si le frag 'std::fixed' est
   * actif ou non pour l'écriture des valeurs.
   * 
   * @return true Si oui.
   * @return false Si non.
   */
  virtual bool isFixed() = 0;
  /**
   * @brief Méthode permettant de définir le flag 'std::fixed' ou non.
   * 
   * Aussi bien pour la méthode 'print()' que la méthode 'writetable()'.
   * 
   * Ce flag permet de 'forcer' le nombre de chiffre après la virgule à
   * la précision voulu. Par exemple, si l'on a appelé 'setPrecision(4)',
   * et que l'on appelle 'setFixed(true)', le print de '6.1' donnera '6.1000'.
   * 
   * @warning Le flag "std::fixed" modifie le comportement de "setPrecision()",
   *          si le flag "std::fixed" est désactivé, la précision définira le
   *          nombre de chiffres total (avant et après la virgule) ;
   *          si le flag "std::fixed" est activé, la précision définira le
   *          nombre de chiffres après la virgule. Attention donc lors de
   *          l'utilisation de "std::numeric_limits<Real>::max_digits10"
   *          (pour l'écriture) ou de "std::numeric_limits<Real>::digits10"
   *          (pour la lecture) qui sont à utiliser sans le flag "std::fixed".
   * 
   * @param fixed Si le flag 'std::fixed' doit être défini ou non.
   */
  virtual void setFixed(bool fixed) = 0;

  /**
   * @brief Méthode permettant de savoir si le frag 'std::scientific' est
   * actif ou non pour l'écriture des valeurs.
   * 
   * @return true Si oui.
   * @return false Si non.
   */
  virtual bool isForcedToUseScientificNotation() = 0;
  /**
   * @brief Méthode permettant de définir le flag 'std::scientific' ou non.
   * 
   * Aussi bien pour la méthode 'print()' que la méthode 'writetable()'.
   * 
   * Ce flag permet de 'forcer' l'affichage des valeurs en écriture
   * scientifique.
   * 
   * @param use_scientific Si le flag 'std::scientific' doit être défini ou non.
   */
  virtual void setForcedToUseScientificNotation(bool use_scientific) = 0;

  /**
   * @brief Accesseur permettant de récupérer le nom du répertoire où sera
   * placé les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le répertoire.
   */
  virtual String outputDirectory() = 0;
  /**
   * @brief Accesseur permettant de définir le répertoire
   * dans lequel enregistrer les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @param directory Le répertoire.
   */
  virtual void setOutputDirectory(const String& directory) = 0;

  /**
   * @brief Accesseur permettant de récupérer le nom des tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le nom.
   */
  virtual String tableName() = 0;
  /**
   * @brief Accesseur permettant de définir le nom du tableau.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @param name Le nom.
   */
  virtual void setTableName(const String& name) = 0;

  /**
   * @brief Accesseur permettant de récupérer le nom des fichiers.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le nom.
   */
  virtual String fileName() = 0;

  /**
   * @brief Accesseur permettant de récupérer le chemin où sera
   * enregistrés les tableaux. 
   * 
   * En comparaison avec rootPathOutput(), le retour peut être
   * different selon le "directory" et le "name".
   * 
   * @return String Le chemin.
   */
  virtual Directory outputPath() = 0;

  /**
   * @brief Accesseur permettant de récupérer le chemin où l'implémentation
   * enregistre ces tableaux. 
   * 
   * En comparaison avec pathOutput(), le retour ne dépend pas de "directory" ou de "name".
   * 
   * @return String Le chemin.
   */
  virtual Directory rootPath() = 0;

  /**
   * @brief Méthode permettant de savoir si les paramètres actuellement en possession
   * de l'implémentation lui permet d'écrire un fichier par processus.
   * 
   * @return true Si oui, l'implémentation peut écrire un fichier par processus.
   * @return false Sinon, il n'y a qu'un seul fichier qui peut être écrit.
   */
  virtual bool isOneFileByRanksPermited() = 0;

  /**
   * @brief Méthode permettant de connaitre le type de fichier du service.
   * 
   * @return String Le type de fichier.
   */
  virtual String fileType() = 0;

  /**
   * @brief Méthode permettant de récupérer une référence vers l'objet
   * SimpleTableInternal utilisé.
   * 
   * @return Ref<SimpleTableInternal> Une copie de la référence. 
   */
  virtual Ref<SimpleTableInternal> internal() = 0;

  /**
   * @brief Méthode permettant de récupérer une référence vers l'objet
   * ISimpleTableReaderWriter utilisé.
   * 
   * @return Ref<ISimpleTableReaderWriter> Une copie de la référence. 
   */
  virtual Ref<ISimpleTableReaderWriter> readerWriter() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
