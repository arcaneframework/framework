// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableWriterHelper.hh                                       (C) 2000-2022 */
/*                                                                           */
/* TODO         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEWRITERHELPER_H
#define ARCANE_ISIMPLETABLEWRITERHELPER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/ItemTypes.h>
#include <arcane/Directory.h>
#include <arcane/ISimpleTableMng.h>
#include <arcane/ISimpleTableReaderWriter.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief TODO
 */
class ARCANE_CORE_EXPORT ISimpleTableWriterHelper
{
public:
  virtual ~ISimpleTableWriterHelper() = default;

public:

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'initialiser le tableau.
   */
  virtual bool init() = 0;
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   * 
   * @param name_table Le nom du tableau (et du fichier de sortie).
   */
  virtual bool init(String name_table) = 0;
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   * 
   * @param name_table Le nom du tableau (et du fichier de sortie).
   * @param name_dir Le nom du dossier dans lequel enregistrer les tableaux.
   */
  virtual bool init(String name_table, String name_dir) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'afficher le tableau.
   * 
   * @param only_proc L'id du processus devant afficher le tableau (-1 pour 
   *                  signifier "tous les processus").
   */
  virtual void print(Integer only_proc = 0) = 0;

  virtual bool writeFile(Directory root_dir, Integer only_proc) = 0;

  /**
   * @brief Méthode permettant d'écrire le tableau dans un fichier.
   * Si only_proc != -1, les processus autres que P0 retournent true.
   * 
   * @param only_proc L'id du processus devant écrire dans un fichier 
   *                  le tableau (-1 pour signifier "tous les processus").
   * @return true Si le fichier a été correctement écrit.
   * @return false Si le fichier n'a pas été correctement écrit.
   */
  virtual bool writeFile(Integer only_proc = -1) = 0;

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
   * \note Un appel à cette méthode sans le paramètre définira la précision
   * par défaut.
   * 
   * @param precision La nouvelle précision.
   */
  virtual void setPrecision(Integer precision = 6) = 0;

  /**
   * @brief Méthode permettant de savoir si le frag 'std::fixed' est
   * actif ou non pour l'écriture des valeurs.
   * 
   * @return true Si oui.
   * @return false Si non.
   */
  virtual bool fixed() = 0;
  /**
   * @brief Méthode permettant de définir le flag 'std::fixed' ou non.
   * 
   * Aussi bien pour la méthode 'print()' que les méthodes 'writeFile()'.
   * 
   * Ce flag permet de 'forcer' le nombre de chiffre après la virgule à
   * la précision voulu. Par exemple, si l'on a appelé 'setPrecision(4)',
   * et que l'on appelle 'setFixed(true)', le print de '6.1' donnera '6.1000'.
   * 
   * \note Un appel à cette méthode sans le paramètre définira le flag
   * par défaut.
   * 
   * @param fixed Si le flag 'std::fixed' doit être défini ou non.
   */
  virtual void setFixed(bool fixed = true) = 0;

  virtual String outputDirWithoutComputation() = 0;

  /**
   * @brief Accesseur permettant de récupérer le nom du répertoire où sera
   * placé les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le répertoire.
   */
  virtual String outputDir() = 0;
  /**
   * @brief Accesseur permettant de définir le répertoire
   * dans lequel enregistrer les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @param dir Le répertoire.
   */
  virtual void setOutputDir(String dir) = 0;

  virtual String tabNameWithoutComputation() = 0;


  /**
   * @brief Accesseur permettant de récupérer le nom des tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le nom.
   */
  virtual String tabName() = 0;
  /**
   * @brief Accesseur permettant de définir le nom du tableau.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @param name Le nom.
   */
  virtual void setTabName(String name) = 0;

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
   * different selon le "dir" et le "name".
   * 
   * @return String Le chemin.
   */
  virtual Directory outputPath() = 0;


  /**
   * @brief Accesseur permettant de récupérer le chemin où l'implémentation
   * enregistre ces tableaux. 
   * 
   * En comparaison avec pathOutput(), le retour ne dépend pas de "dir" ou de "name".
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
  virtual bool isOneFileByProcsPermited() = 0;

  virtual String typeFile() = 0;

  virtual SimpleTableInternal* internal() = 0;

  virtual ISimpleTableReaderWriter* readerWriter() = 0;
  virtual void setReaderWriter(ISimpleTableReaderWriter* strw) = 0;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
