// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaCutInfosReader.cc                                       (C) 2000-2018 */
/*                                                                           */
/* Lecteur des informations de découpages avec les fichiers Lima.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/ItemGroup.h"
#include "arcane/Item.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"
#include "arcane/IParallelMng.h"
#include "arcane/IIOMng.h"
#include "arcane/Timer.h"
#include "arcane/IMesh.h"

#include "arcane/cea/LimaCutInfosReader.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

//Juste pour faire le .lib sous Windows.
class ARCANE_EXPORT LimaTest
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LimaCutInfosReader::
LimaCutInfosReader(IParallelMng* parallel_mng)
: TraceAccessor(parallel_mng->traceMng())
, m_parallel_mng(parallel_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des correspondances.
 *
 * Les correspondances sont stockées en mémoire à l'adresse \a buf sous la
 * forme d'une suite d'entier au format ascii séparés par des espaces.
 * Par exemple: "125 132 256".
 */
static Integer
_readList(Int64ArrayView& int_list,const char* buf)
{
  istringstream istr(buf);
  Integer index = 0;
  while (!istr.eof()){
    Int64 v = 0;
    istr >> v;
    if (istr.eof())
      break;
    int_list[index] = v;
    ++index;
  }
  return index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construit les structures internes du maillage.
 */
void LimaCutInfosReader::
readItemsUniqueId(Int64ArrayView nodes_id,Int64ArrayView cells_id,
                  const String& dir_name)
{
  _readUniqueIndex(nodes_id,cells_id,dir_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupération des indices uniques des entités.
 *
 * Récupère pour chaque entité son numéro unique pour tout domaine.
 * Les valeurs sont stockées dans \a nodes_id pour les noeuds et
 * \a cells_id pour les mailles. Ces deux tableaux doivent déjà avoir été
 * alloués à la bonne taille.

 * Le fonctionnement est le suivant:
 * \arg le processeur 0 lit le fichier de correspondance et récupère les
 * informations qui le concerne.
 * \arg pour chaque autre processeur, le processeur 0 récupère le nombre
 * de noeuds et mailles à lire, lit les valeurs dans le fichier de
 * correspondance et les transfert au processeur.
 */
void LimaCutInfosReader::
_readUniqueIndex(Int64ArrayView nodes_id,Int64ArrayView cells_id,
                 const String& dir_name)
{
  Timer time_to_read(m_parallel_mng->timerMng(),"ReadCorrespondance",Timer::TimerReal);
  IParallelMng* pm = m_parallel_mng;

  // Si le cas est parallèle, lecture du fichier de correspondance.
  bool is_parallel   = pm->isParallel();
  Int32 comm_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  // Si on est dans le cas où Arcane est retranché à un coeur
  if ((!is_parallel) || (is_parallel && nb_rank==1)){
    for( Integer i=0, n=nodes_id.size(); i<n; ++i )
      nodes_id[i] = i;
    for( Integer i=0, n=cells_id.size(); i<n; ++i )
      cells_id[i] = i;
    return;
  }

  ScopedPtrT<IXmlDocumentHolder> doc_holder;
  StringBuilder correspondance_filename;
  if (!dir_name.empty()){
    correspondance_filename += dir_name;
    correspondance_filename += "/";
  }
  correspondance_filename += "Correspondances";

  if (comm_rank==0){
    {
      Timer::Sentry sentry(&time_to_read);
      IXmlDocumentHolder* doc = pm->ioMng()->parseXmlFile(correspondance_filename);
      doc_holder = doc;
    }
    if (!doc_holder.get())
      ARCANE_FATAL("Invalid correspondance file '{0}'",correspondance_filename);

    info() << "Time to read (unit: second) 'Correspondances' file: "
           << time_to_read.lastActivationTime();

    XmlNode root_element = doc_holder->documentNode().documentElement();
    
    // D'abord, le sous-domaine lit ses valeurs.
    _readUniqueIndexFromXml(nodes_id,cells_id,root_element,0);
    
    {
      Timer::Sentry sentry(&time_to_read);
      // Ensuite boucle, sur les autres sous-domaine.
      Int64UniqueArray other_nodes_id;
      Int64UniqueArray other_cells_id;
      UniqueArray<Integer> other_sizes(2);
      for( Int32 i=1; i<nb_rank; ++i ){
        pm->recv(other_sizes,i);
        other_nodes_id.resize(other_sizes[0]);
        other_cells_id.resize(other_sizes[1]);
        _readUniqueIndexFromXml(other_nodes_id,other_cells_id,root_element,i);
        pm->send(other_nodes_id,i);
        pm->send(other_cells_id,i);
      }
    }
    info() << "Time to transfert values: "
           << time_to_read.lastActivationTime();
  }
  else{
    // Réceptionne les valeurs envoyées par le sous-domaine 0
    Integer mes[2];
    IntegerArrayView mesh_element_size(2,mes);
    mesh_element_size[0] = nodes_id.size();
    mesh_element_size[1] = cells_id.size();
    pm->send(mesh_element_size,0);
    pm->recv(nodes_id,0);
    pm->recv(cells_id,0);
  }
  // Vérifie que tous les messages sont envoyés et recus avant de détruire
  // le document XML.
  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des indices uniques à partir d'un fichier de correspondance
 * XML.
 * \param nodes_id indices des noeuds
 * \param cells_id indices des mailles
 * \param root_element élément racine de l'arbre XML.
 * \param comm_rank numéro du sous-domaine à lire.
 */
void LimaCutInfosReader::
_readUniqueIndexFromXml(Int64ArrayView nodes_id,Int64ArrayView cells_id,
                        XmlNode root_element,Int32 comm_rank)
{
  XmlNode cpu_elem;
  XmlNodeList cpu_list = root_element.children(String("cpu"));

  String ustr_buf(String::fromNumber(comm_rank));

  String us_id("id");
  for( Integer i=0, s=cpu_list.size(); i<s; ++i ){
    String id_str = cpu_list[i].attrValue(us_id);
    if (id_str.null())
      continue;
    if (id_str==ustr_buf){
      cpu_elem = cpu_list[i];
      break;
    }
  }
  if (cpu_elem.null())
    ARCANE_FATAL("No element <cpu[@id=\"{0}\"]>",comm_rank);
  XmlNode node_elem = cpu_elem.child("noeuds");
  if (node_elem.null())
    ARCANE_FATAL("No element <noeuds>");
  XmlNode cell_elem = cpu_elem.child("mailles");
  if (cell_elem.null())
    ARCANE_FATAL("No element <mailles>");
  
  // Tableau de correspodance des noeuds
  {
    String ustr_value = node_elem.value();
    Integer nb_read = _readList(nodes_id,ustr_value.localstr());
    Integer expected_size = nodes_id.size();
    if (nb_read!=expected_size)
      ARCANE_FATAL("Bad number of nodes rank={0} nb_read={1} expected={2}",
                   comm_rank,nb_read,expected_size);
  }
  
  // Tableau de correspodance des mailles
  {
    String ustr_value = cell_elem.value();
    Integer nb_read = _readList(cells_id,ustr_value.localstr());
    Integer expected_size = cells_id.size();
    if (nb_read!=expected_size)
      ARCANE_FATAL("Bad number of cells rank={0} nb_read={1} expected={2}",
                   comm_rank,nb_read,expected_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

