// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableWriterHelper.h                                  (C) 2000-2022 */
/*                                                                           */
/* Interface représentant un écrivain simple utilisant un                    */
/* ISimpleTableReaderWriter.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEWRITERHELPER_H
#define ARCANE_ISIMPLETABLEWRITERHELPER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/ItemTypes.h>
#include <arcane/Directory.h>
#include <arcane/ISimpleTableInternalMng.h>
#include <arcane/ISimpleTableReaderWriter.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Interface de classe permettant d'écrire un fichier
 * avec ISimpleTableReaderWriter.
 * Fournit des méthodes permettant de gérer l'écriture en parallèle
 * et les symboles de noms.
 * 
 * Cette classe est, en quelque sorte, une surcouche de 
 * ISimpleTableReaderWriter qui est assez basique.
 * ISimpleTableWriterHelper est là pour simplifier
 * l'utilisation de ISimpleTableReaderWriter.
 * 
 * Dans la partie SimpleTable, les symboles de noms sont des
 * mots-clefs entourés d'arobases et qui seront replacés
 * par leur signfication lors de l'exécution.
 * Dans l'implémentation SimpleTableWriterHelper, il y a
 * actuellement deux symboles de noms pris en charge :
 * - \@proc_id\@ : Sera remplacé par l'id du processus.
 * - \@num_procs\@ : Sera remplacé par le nombre de processus.
 * Et dans SimpleTableWriterHelper, ces symboles ne sont remplacés
 * que dans le nom du tableau.
 */
class ARCANE_CORE_EXPORT ISimpleTableWriterHelper
{
public:
  virtual ~ISimpleTableWriterHelper() = default;

public:

  /**
   * @brief Méthode permettant d'initialiser l'objet.
   * Notamment le nom du tableau (name_tab) et le nom du répertoire qui
   * contiendra les fichiers (le répertoire des tableaux/name_dir).
   * 
   * Les valeurs par défauts de name_tab et de name_dir sont
   * laissées à la discretion de l'implémentation.
   */
  virtual bool init() = 0;
  /**
   * @brief Méthode permettant d'initialiser l'objet.
   * Notamment le nom du tableau et le nom du répertoire qui contiendra
   * les fichiers (le répertoire des tableaux/name_dir).
   * 
   * La valeur par défaut de name_dir est laissée à la discretion 
   * de l'implémentation.
   * 
   * @param name_table Le nom du tableau (et du fichier de sortie).
   */
  virtual bool init(String name_table) = 0;
  /**
   * @brief Méthode permettant d'initialiser l'objet.
   * Notamment le nom du tableau et le nom du répertoire qui contiendra
   * les fichiers (le répertoire des tableaux/name_dir).
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

  /**
   * @brief Méthode permettant d'écrire le tableau dans un fichier.
   * Si only_proc != -1, les processus autres que only_proc retournent true.
   * 
   * Par exemple, dans l'implémentation SimpleTableWriterHelper,
   * le ou les fichiers seront écrits dans le répertoire :
   * root_dir/[name_dir]/[name_table].[ISimpleTableReaderWriter.typeFile()]
   * 
   * @param root_dir Le répertoire racine où créer le répertoire des tableaux.
   * @param only_proc L'id du processus devant écrire dans un fichier 
   *                  le tableau (-1 pour signifier "tous les processus").
   * @return true Si le fichier a été correctement écrit.
   * @return false Si le fichier n'a pas été correctement écrit.
   */
  virtual bool writeFile(Directory root_dir, Integer only_proc) = 0;

  /**
   * @brief Méthode permettant d'écrire le tableau dans un fichier.
   * Si only_proc != -1, les processus autres que only_proc retournent true.
   * 
   * Par exemple, dans l'implémentation SimpleTableWriterHelper,
   * le ou les fichiers seront écrits dans le répertoire :
   * ./[output]/[name_dir]/[name_table].[ISimpleTableReaderWriter.typeFile()]
   * "output" est fourni par Arcane.
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
   * @note Un appel à cette méthode sans le paramètre définira la précision
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

  /**
   * @brief Méthode permettant de récupérer le nom du répertoire tel qu'il
   * a été donné précédement.
   * 
   * Ici, les symboles de noms sont toujours présent.
   * 
   * @return String Le répertoire.
   */
  virtual String outputDirWithoutComputation() = 0;

  /**
   * @brief Méthode permettant de récupérer le nom du répertoire où sera
   * placé les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * Ici, les symboles de noms ont été résolus.
   * 
   * @return String Le répertoire.
   */
  virtual String outputDir() = 0;
  /**
   * @brief Méthode permettant de définir le répertoire
   * dans lequel enregistrer les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @param dir Le répertoire.
   */
  virtual void setOutputDir(String dir) = 0;


  /**
   * @brief Méthode permettant de récupérer le nom du tableau tel qu'il
   * a été donné précédement.
   * 
   * Ici, les symboles de noms sont toujours présent.
   * 
   * @return String Le nom.
   */
  virtual String tabNameWithoutComputation() = 0;

  /**
   * @brief Méthode permettant de récupérer le nom du tableau.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * Ici, les symboles de noms ont été résolus.
   * 
   * @return String Le nom.
   */
  virtual String tabName() = 0;
  /**
   * @brief Méthode permettant de définir le nom du tableau.
   * 
   * @param name Le nom.
   */
  virtual void setTabName(String name) = 0;

  /**
   * @brief Méthode permettant de récupérer le nom du fichier.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * Ici, les symboles de noms ont été résolus et l'extension est ajoutée.
   * 
   * @return String Le nom.
   */
  virtual String fileName() = 0;

  /**
   * @brief Méthode permettant de récupérer le chemin où sera
   * enregistrés les tableaux. 
   * 
   * Exemple (relatif) :
   * ./output/csv/[name_dir]/
   * 
   * @return String Le chemin.
   */
  virtual Directory outputPath() = 0;


  /**
   * @brief Méthode permettant de récupérer le chemin où l'implémentation
   * enregistre ces tableaux. 
   * 
   * Exemple (relatif) :
   * ./output/csv/
   * 
   * @return String Le chemin.
   */
  virtual Directory rootPath() = 0;

  /**
   * @brief Méthode permettant de savoir si les paramètres actuellement en possession
   * de l'implémentation lui permet d'écrire un fichier par processus, notamment 
   * grâce aux symboles de noms.
   * 
   * @return true Si oui, l'implémentation peut écrire un fichier par processus.
   * @return false Sinon, il n'y a qu'un seul fichier qui peut être écrit.
   */
  virtual bool isOneFileByProcsPermited() = 0;

  /**
   * @brief Méthode permettant de connaitre le type de fichier qui sera utilisé.
   * 
   * @return String 
   */
  virtual String typeFile() = 0;

  /**
   * @brief Méthode permettant de récupérer le pointeur vers l'objet
   * SimpleTableInternal utilisé.
   * 
   * @return SimpleTableInternal* Le pointeur utilisé. 
   */
  virtual SimpleTableInternal* internal() = 0;

  /**
   * @brief Méthode permettant de récupérer le pointeur vers l'objet
   * ISimpleTableReaderWriter utilisé.
   * 
   * @return ISimpleTableReaderWriter* Le pointeur utilisé.
   */
  virtual ISimpleTableReaderWriter* readerWriter() = 0;

  /**
   * @brief Méthode permettant de définir un pointeur vers
   * une implem de ISimpleTableReaderWriter.
   * 
   * @warning Il est déconseillé d'utiliser cette méthode, sauf si
   * vous savez ce que vous faite. La destruction de l'objet reste
   * à la charge de l'appelant.
   * 
   * @param strw Le pointeur vers une implem de ISimpleTableReaderWriter.
   */
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
