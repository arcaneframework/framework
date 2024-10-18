//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlInfo.h                                                   (C) 2000-2024 */
/*                                                                           */
/* Informations sur un fichier XML.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;
using System.Xml;
using System.Xml.Schema;

namespace Arcane.Axl
{
  public class MyXmlResolver : XmlUrlResolver
  {
    public override object GetEntity(Uri uri,string name,Type type)
    {
      if(GlobalContext.Instance.Verbose){ 
        Console.WriteLine("RESOLVE: name={0} uri={1}",name,uri);
      }
      return base.GetEntity(uri,name,type);
    }
    
  }

  /*!
   * \internal
   * \brief Informations sur un fichier XML.
   *
   * Classe de base des classes stockant les informations lues dans les
   * fichiers XML. Un élément XML nommé XXX aura ses informations stockées
   * dans une classe XXXInfo.
   */
  public class XmlInfo
  {
    public XmlInfo() { }

    /**
      * Leve une exception concernant l'élément XML "element" avec le message "msg".
      * @param element élément XML concerné par l'exception.
      * @param msg message de l'exception.
      */
    public static void Error(XmlNode node,string msg)
    {
      string s = "** Erreur: Elément <" + node.Name + ">: " + msg + '.';
      throw new InternalErrorException(s);
    }

    /**
     * Affiche le message d'avertissement "msg" concernant l'élément XML "element".
     * @param element élément XML concerné par l'exception.
     * @param msg message de l'exception.
     */
    public static void Warning(XmlNode node, string msg)
    {
      Console.WriteLine("** Warning: Elément <{0}>: {1}.\n", node.Name, msg);
    }

    /**
     * Leve une exception concernant l'élément XML "element" en précisant que
     * l'attribut "attr_name" a été oublié dans le fichier XML.
     * @param element élément XML concerné par l'exception.
     * @param attr_name nom de l'attribut concerné.
     */
    public static void AttrError(XmlElement element, string attr_name)
    {
      string s = "Attribut \"" + attr_name + "\" non spécifié";
      Error(element, s);
    }

    /**
     * Retourne le noeud racine du fichier XML nommé "file_name". Si un nom
     * de schéma est passé en argument, la conformité du fichier au schéma
     * est vérifié.
     * @return le noeud XML racine du document.
     * @param file_name le nom du fichier XML.
     * @param mng le gestionnaire d'entrées/sorties Arcane.
     * @param schema_name le nom du schéma associé au fichier (optionnel).
     */
    public static XmlElement RootNode(string file_name)
    {
      XmlDocument doc = new XmlDocument();
      doc.Load(file_name);
      return doc.DocumentElement;
    }

    /**
     * Retourne le noeud racine du fichier XML nommé "file_name". Si un nom
     * de schéma est passé en argument, la conformité du fichier au schéma
     * est vérifié.
     * @return le noeud XML racine du document.
     * @param file_name le nom du fichier XML.
     * @param mng le gestionnaire d'entrées/sorties Arcane.
     * @param schema_stream le flux du schéma associé au fichier (optionnel).
     */
    public static XmlElement RootNode(string file_name, Stream schema_stream)
    {
      // Note: avec la version de mono apres 1.1.16.1, la validation ne fonctionne pas
      // à cause d'un problème de namespace. On la désactive donc.
      XmlReaderSettings settings = new XmlReaderSettings();
      settings.XmlResolver = new MyXmlResolver();
      // Autorise la lecture des DtD.
      settings.DtdProcessing = DtdProcessing.Parse;

      bool do_validation = false;
      if (do_validation){
        XmlSchemaSet sc = new XmlSchemaSet();
        //sc.Add("urn:arcane-axl",new XmlTextReader(schema_stream));
        sc.Add(null,new XmlTextReader(schema_stream));
        //sc.Add (XmlSchema.Read (schema_stream, null));
        settings.ValidationType = ValidationType.Schema;
        settings.Schemas = sc;
      }
      else if(GlobalContext.Instance.Verbose){
        Console.WriteLine("ATTENTION: validation avec axl.xsd désactivée");
      }
      XmlNameTable name_table = new NameTable();
      string subset = _GetAxlEntities();
      XmlParserContext context = new XmlParserContext(name_table,null,"axl",null,null,subset,"","",XmlSpace.Preserve);

      XmlReader reader = XmlReader.Create(file_name,settings,context);
      //reader.ExpandEntities = false;
      //reader.EntityHandling = EntityHandling.
      //while (reader.Read());

      XmlDocument doc = new XmlDocument();
      doc.Load(reader);
      return doc.DocumentElement;
    }

    static object m_entities_lock = new object();
    static string m_entities;
    static string _GetAxlEntities()
    {
      lock(m_entities_lock){
        if (m_entities!=null)
          return m_entities;
        m_entities = CommonEntities.Build();
      }
      return m_entities;
    }
  }
}
