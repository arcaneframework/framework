// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIOMng.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des entrées-sorties.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IIOMNG_H
#define ARCANE_CORE_IIOMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 * \brief Interface du gestionnaire des entrées sorties.
 *
 * \todo gestionnaire des entrées sorties permettant d'encapsuler
 * la gestion des fichiers en parallèles.
 */
class ARCANE_CORE_EXPORT IIOMng
{
 public:

  virtual ~IIOMng() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Lit et analyse le fichier XML \a filename.
   *
   * En cas d'erreur, retourne 0.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   * Si un nom de schéma est spécifié, la cohérence
   * du fichier relativement au schéma est vérifiée.
   */
  virtual IXmlDocumentHolder*
  parseXmlFile(const String& filename, const String& schemaname = String{}) = 0;

  /*!
   * \brief Lit et analyse le fichier XML \a filename.
   *
   * En cas d'erreur, retourne 0.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   * La cohérence du fichier relativement au schéma est vérifiée; 
   * Le nom du schéma est donnée uniquement pour traitement des messages d'erreurs.
   */
  virtual IXmlDocumentHolder* parseXmlFile(const String& filename,
                                           const String& schemaname,
                                           ConstArrayView<Byte> schema_data) = 0;

  /*!
   * \brief Lit et analyse le fichier XML contenu dans le buffer \a buffer.
   *
   * En cas d'erreur, retourne 0.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   * L'argument \a name associe un nom à la zone mémoire qui est
   * utilisé pour afficher les messages d'erreur.
   */
  virtual IXmlDocumentHolder* parseXmlBuffer(Span<const Byte> buffer, const String& name) = 0;

  /*!
   * \brief Lit et analyse le fichier XML contenu dans le buffer \a buffer.
   *
   * En cas d'erreur, retourne 0.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   * L'argument \a name associe un nom à la zone mémoire qui est
   * utilisé pour afficher les messages d'erreur.
   */
  virtual IXmlDocumentHolder* parseXmlBuffer(Span<const std::byte> buffer, const String& name) = 0;

  /*!
   * \brief Lit et analyse le fichier XML contenu dans la chaîne \a str.
   *
   * En cas d'erreur, retourne 0.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   * L'argument \a name associe un nom à la zone mémoire qui est
   * utilisé pour afficher les messages d'erreur.
   */
  virtual IXmlDocumentHolder* parseXmlString(const String& str, const String& name) = 0;

  /*! \brief Ecrit l'arbre XML du document \a doc dans le fichier filename.
   * \retval true en cas d'erreur,
   * \return false en cas de succès.
   */
  virtual bool writeXmlFile(IXmlDocumentHolder* doc, const String& filename, const bool indented = false) = 0;

  /*!
   * \brief Lecture collective d'un fichier.
   *
   * Lit collectivement le fichier \a filename et retourne son
   * contenu dans \a bytes. Le fichier est considéré comme un fichier binaire.
   * La lecture collective signifie que l'ensemble des
   * processeurs appellent cette opération et vont lire le même fichier.
   * L'implémentation peut alors optimiser les accès disque en regroupant la
   * lecture effective sur un ou plusieurs processeurs puis envoyer le
   * contenu du fichier sur les autres.
   *
   * \retval true en cas d'erreur
   * \retval false si tout est ok.
   */
  virtual bool collectiveRead(const String& filename, ByteArray& bytes) = 0;

  /*!
   * \brief Lecture collective d'un fichier.
   *
   * Lit collectivement le fichier \a filename et retourne son
   * contenu dans \a bytes. Le fichier est considéré comme un fichier binaire
   * si \a is_binary est vrai.
   * La lecture collective signifie que l'ensemble des
   * processeurs appellent cette opération et vont lire le même fichier.
   * L'implémentation peut alors optimiser les accès disque en regroupant la
   * lecture effective sur un ou plusieurs processeurs puis envoyer le
   * contenu du fichier sur les autres.
   *
   * \retval true en cas d'erreur
   * \retval false si tout est ok.
   */
  virtual bool collectiveRead(const String& filename, ByteArray& bytes, bool is_binary) = 0;

  /*!
   * \brief Lecture locale d'un fichier.
   *
   * Lit localement le fichier \a filename et retourne son
   * contenu dans \a bytes. Le fichier est considéré comme un fichier binaire.
   * Cette opération n'est pas collective.
   *
   * \retval true en cas d'erreur.
   * \retval false si tout est ok.
   *
   * \warning retourne aussi true si le fichier est vide.
   * \warning si le ByteUniqueArray doit être converti en String, il _faut_ y ajouter un 0 terminal au préalable (bytes.add(0))
   */
  virtual bool localRead(const String& filename, ByteArray& bytes) = 0;

  /*!
   * \brief Lecture locale d'un fichier.
   *
   * Lit localement le fichier \a filename et retourne son
   * contenu dans \a bytes.
   * Cette opération n'est pas collective.
   *
   * \retval true en cas d'erreur.
   * \retval false si tout est ok.
   *
   * \warning retourne aussi true si le fichier est vide.
   * \warning si le ByteUniqueArray doit être converti en String, il _faut_ y ajouter un 0 terminal au préalable (bytes.add(0))
   */
  virtual bool localRead(const String& filename, ByteArray& bytes, bool is_binary) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

