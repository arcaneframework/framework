//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoxygenDocumentationVisitor.cs                              (C) 2000-2018 */
/*                                                                           */
/* Génération de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Linq;
using Arcane.Axl;
using Arcane.AxlDoc.UserInterfaces;

namespace Arcane.AxlDoc
{
  public class DoxygenDocumentationVisitor : IOptionInfoVisitor
  {
    private TextWriter m_brief_stream;
    private TextWriter m_full_stream;
    private DoxygenDocumentationFile m_doc_file;
    private CodeInfo m_code_info;
    private bool m_print_user_class;
    private StreamWriter m_dico_writer = null;
    private Config m_config;
    private List<IExtraDescriptionWriter> m_extra_description_writers; // Allows to add info in description subnode

    public void AddExtraDescriptionWriter (IExtraDescriptionWriter extra_description_writer)
    {
      m_extra_description_writers.Add (extra_description_writer);
    }

    public bool PrintUserClass { get { return m_print_user_class; } set { m_print_user_class = value; } }

    public DoxygenDocumentationVisitor (DoxygenDocumentationFile doc_file, 
                                        CodeInfo code_info, 
                                        StreamWriter dico_writer, 
                                        Config config)
    {
      m_full_stream = doc_file.FullDescStream;
      m_brief_stream = doc_file.BriefDescStream;
      m_doc_file = doc_file;
      m_code_info = code_info;
      m_dico_writer = dico_writer;
      m_config = config;
      m_extra_description_writers = new List<IExtraDescriptionWriter> ();
    }

    public void VisitServiceOrModule (ServiceOrModuleInfo info)
    {
      ServiceInfo sinfo = info as ServiceInfo;
      if (sinfo != null) {
        StringBuilder sb = new StringBuilder ();

        sb.Append ("\n## Interfaces\n");

        foreach (ServiceInfo.Interface ii in sinfo.Interfaces) {
          // TODO TRAD : Si traduction FR/EN : A traduire
          // sb.AppendFormat ("- Implémente l'interface \\ref axldoc_interface_{1} \"{0}\"\n", ii.Name, DoxygenDocumentationUtils.AnchorName(ii.Name));
          sb.AppendFormat ("- Implements \\ref axldoc_interface_{1} \"{0}\" interface\n", ii.Name, DoxygenDocumentationUtils.AnchorName(ii.Name));
        }
        // Write hyperlinks to module or service using the service
        CodeInterfaceInfo interface_info = null;
        if (m_code_info.Interfaces.TryGetValue (sinfo.Type, out interface_info) && interface_info.Users.Count>0) {
          sb.AppendFormat ("<dl>");
          // TODO TRAD : Si traduction FR/EN : A traduire
          // sb.AppendFormat ("<dt>Service utilisé dans</dt>");
          sb.AppendFormat ("<dt>Service used in</dt>");
          foreach (CodeInterfaceUserInfo user_info in interface_info.Users) {
            sb.AppendFormat ("<dd>\\ref axldoc_{0} \"{1}\" {2}<br/></dd>", user_info.Hyperlink, user_info.Name, (user_info.IsModule)?"module":"service");
          }
          sb.AppendFormat ("</dl>");
        }
        m_doc_file.SubTitle = sb.ToString ();
      }

      _WritePageFullDesc (null, info.DescriptionElement);
      // Pour retrouver facilement une option dans la doc, on la trie
      // pas ordre alphabetique (si m_do_sort est activé)
      List<Option> not_sorted_options = new List<Option> ();
      foreach (Option o in info.Options) {
        if (m_config.private_app_pages.Filter (o)) {
          string name = o.GetTranslatedName (m_code_info.Language);
          // Attention, il est possible par erreur que deux options aient le mÃªme nom traduit.
          // Pour corriger cette erreur spécifique peut être générée
          if (not_sorted_options.Exists(op => op.GetTranslatedName(m_code_info.Language) == name)) {
            throw new ApplicationException (String.Format ("L'option {0} a la même la traduction '{1}' qu'une autre option de même niveau hiérarchique",
                                                           o.GetIdString (), name));
          }
          not_sorted_options.Add (o);
        }
      }

      if (m_config.do_sort != SortMode.None) {
        not_sorted_options.Sort((a,b) => String.Compare(a.GetTranslatedName (m_code_info.Language), 
                                                        b.GetTranslatedName (m_code_info.Language)));
      }

      if (not_sorted_options.Count > 0) {
        foreach (Option o in not_sorted_options) {
          o.Accept (this);
        }
      }
    }

    public void VisitComplex (ComplexOptionInfo o)
    {
      OptionTypeCounterVisitor otc = new OptionTypeCounterVisitor ();
      o.AcceptChildren (otc, x => m_config.private_app_pages.Filter(x));

      if (m_config.max_display_size > 0 && otc.NbTotalOption > m_config.max_display_size) {
        _AddBriefDescription (o, true);
        DoxygenDocumentationFile df = new DoxygenDocumentationFile (o, m_doc_file.OutputPath, m_code_info.Language);
        DoxygenDocumentationVisitor dv = new DoxygenDocumentationVisitor (df, m_code_info, m_dico_writer, m_config);
        // Two choices: same extra description writers or new instances ? Here, same.
        foreach(var dw in m_extra_description_writers)
          dv.AddExtraDescriptionWriter(dw);

        dv._WritePageFullDesc (o, o.DescriptionElement);
        if (m_config.do_sort != SortMode.None)
          o.AcceptChildrenSorted (dv, "fr", x => m_config.private_app_pages.Filter(x));
        else
          o.AcceptChildren (dv, x => m_config.private_app_pages.Filter(x));
        df.Write ();
      } else {
        _WriteHtmlOnly(m_full_stream, "<div class=\"ComplexOptionInfoBlock\">");

        // La partie "détail des méthodes" de doxygen se compose de trois parties :
        // - Un titre (h2 de classe .memtitle)
        // - Une partie "sous-titre" (div de classe .memitem.memproto)
        // - Une partie "description" (div de classe .memitem.memdoc)
        _WriteHtmlOnly(m_full_stream, "<h2 class=\"memtitle\" style=\"border-color: purple;\">");

        _WriteColoredTitle ("purple", o);

        _WriteHtmlOnly(m_full_stream, "</h2>");
        _WriteHtmlOnly(m_full_stream, "<div class=\"memitem\" style=\"border-color: purple;\">");
        _WriteHtmlOnly(m_full_stream, "<div class=\"memproto\">");

        _AddBriefDescription(o, false);
        _AddFullDescription (o);

        _WriteHtmlOnly(m_full_stream, "</div>");
        _WriteHtmlOnly(m_full_stream, "<div class=\"memdoc\">");
        if (otc.NbTotalOption > 0) {
          m_brief_stream.WriteLine ("<ul>");
          if (m_config.do_sort != SortMode.None)
            o.AcceptChildrenSorted (this, "fr", x => m_config.private_app_pages.Filter(x));
          else
            o.AcceptChildren (this, x => m_config.private_app_pages.Filter(x));
          m_brief_stream.WriteLine ("</ul>");
        }
        _WriteCode(o);
        _WriteHtmlOnly(m_full_stream, "</div>");
        _WriteHtmlOnly(m_full_stream, "</div>");
        _WriteHtmlOnly(m_full_stream, "</div><!-- End of ComplexOptionInfoBlock -->"); // Pour ComplexOptionInfoBlock
      }
    }

    public void VisitEnumeration (EnumerationOptionInfo o)
    {
      //_WriteHRw30Tag(m_full_stream);
      // La partie "détail des méthodes" de doxygen se compose de trois parties :
      // - Un titre (h2 de classe .memtitle)
      // - Une partie "sous-titre" (div de classe .memitem.memproto)
      // - Une partie "description" (div de classe .memitem.memdoc)

      m_full_stream.WriteLine ("<h2 class=\"memtitle\" style=\"border-color: olive;\">");
      _WriteColoredTitle ("olive", o);
      m_full_stream.WriteLine ("</h2>");
      XmlDocument owner_doc = o.Node.OwnerDocument;
      XmlElement desc_elem = o.DescriptionElement;
      // Construit dans \a enum_list_element
      // le tableau contenant les descriptions des diffÃ©rentes valeurs
      // que peut prendre l'énumération.
      // Ce tableau est ensuite insérer dans la description, à l'endroit
      // où se trouve l'élément <enum-description> ou s'il est absent,
      // après la description brêve.
      XmlElement enum_list_element = owner_doc.CreateElement ("div");
      enum_list_element.SetAttribute ("class", "EnumTable");
      XmlElement table_element = owner_doc.CreateElement ("table");
      enum_list_element.AppendChild (table_element);
      {
        // Affiche les titres
        XmlElement tr = owner_doc.CreateElement ("tr");
        table_element.AppendChild (tr);
        XmlElement th = owner_doc.CreateElement ("th");
        tr.AppendChild (th);
        string field_name = Utils.XmlGetAttributeValue (desc_elem, "field-name");
        if (field_name == null)
          field_name = "value";
        th.InnerText = field_name;
        th = owner_doc.CreateElement ("th");
        tr.AppendChild (th);
        th.InnerText = "description";
      }
      foreach (EnumValueOptionInfo ev in o.EnumValues) {
        XmlElement tr = owner_doc.CreateElement ("tr");
        table_element.AppendChild (tr);
        XmlElement td1 = owner_doc.CreateElement ("td");
        tr.AppendChild (td1);

        td1.InnerText = ev.GetTranslatedName (m_code_info.Language);

        XmlElement td2 = owner_doc.CreateElement ("td");
        XmlElement sub_desc_elem = ev.DescriptionElement;
        if (sub_desc_elem != null) {
          XmlNodeList children = sub_desc_elem.ChildNodes;
          foreach (XmlNode child in children) {
            td2.AppendChild (child.CloneNode (true));
          }
        }
        tr.AppendChild (td2);
      }
      // enum_list_element.AppendChild (owner_doc.CreateElement ("br"));
      m_full_stream.WriteLine ("<div class=\"memitem\" style=\"border-color: olive;\">");
      m_full_stream.WriteLine ("<div class=\"memproto\">");
      _AddBriefDescription (o, false);
      if (desc_elem != null) {
        XmlElement elem_description = desc_elem.SelectSingleNode ("enum-description") as XmlElement;
        if (elem_description != null) {
          Console.WriteLine ("HAS ELEMENT DESCRIPTION {0} {1}", o.Name, enum_list_element.OuterXml);
          XmlNode desc_parent = elem_description.ParentNode;
          desc_parent.ReplaceChild (enum_list_element, elem_description);
        } else {
          desc_elem.PrependChild (enum_list_element); 
         }
      }
      //m_full_stream.Write(enum_list_element.OuterXml);
      _AddFullDescription (o);
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("<div class=\"memdoc\">");
      _WriteCode(o);
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("</div>");
    }

    public void VisitExtended (ExtendedOptionInfo o)
    {
      // La partie "détail des méthodes" de doxygen se compose de trois parties :
      // - Un titre (h2 de classe .memtitle)
      // - Une partie "sous-titre" (div de classe .memitem.memproto)
      // - Une partie "description" (div de classe .memitem.memdoc)
      m_full_stream.WriteLine ("<h2 class=\"memtitle\" style=\"border-color: teal;\">");
      _WriteColoredTitle ("teal", o);
      m_full_stream.WriteLine ("</h2>");

      m_full_stream.WriteLine ("<div class=\"memitem\" style=\"border-color: teal;\">");
      m_full_stream.WriteLine ("<div class=\"memproto\">");
      _AddBriefDescription (o, false);
      _AddFullDescription (o);
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("<div class=\"memdoc\">");
      _WriteCode(o);
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("</div>");
    }

    public void VisitScript (ScriptOptionInfo o)
    {
      // La partie "détail des méthodes" de doxygen se compose de trois parties :
      // - Un titre (h2 de classe .memtitle)
      // - Une partie "sous-titre" (div de classe .memitem.memproto)
      // - Une partie "description" (div de classe .memitem.memdoc)
      m_full_stream.WriteLine ("<h2 class=\"memtitle\" style=\"border-color: teal;\">");
      _WriteColoredTitle ("teal", o);
      m_full_stream.WriteLine ("</h2>");

      m_full_stream.WriteLine ("<div class=\"memitem\" style=\"border-color: teal;\">");
      m_full_stream.WriteLine ("<div class=\"memproto\">");
      _AddBriefDescription (o, false);
      _AddFullDescription (o);
      m_full_stream.WriteLine ("</div>");

      m_full_stream.WriteLine ("<div class=\"memdoc\">");
      _WriteCode(o);
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("</div>");
    }

    public void VisitSimple (SimpleOptionInfo o)
    {
      // La partie "détail des méthodes" de doxygen se compose de trois parties :
      // - Un titre (h2 de classe .memtitle)
      // - Une partie "sous-titre" (div de classe .memitem.memproto)
      // - Une partie "description" (div de classe .memitem.memdoc)
      m_full_stream.WriteLine ("<h2 class=\"memtitle\" style=\"border-color: green;\">");
      _WriteColoredTitle ("green", o);
      m_full_stream.WriteLine ("</h2>");

      m_full_stream.WriteLine ("<div class=\"memitem\" style=\"border-color: green;\">");
      m_full_stream.WriteLine ("<div class=\"memproto\">");
      // m_full_stream.WriteLine("<p>TYPE={0}</p>",o.Type);
      // if (o.DefaultValue!=null)
      //   m_full_stream.WriteLine("<p>DEFAULT={0}</p>",o.DefaultValue);
      _AddBriefDescription (o, false);
      _AddFullDescription (o);
      m_full_stream.WriteLine ("</div>");

      m_full_stream.WriteLine ("<div class=\"memdoc\">");
      _WriteCode(o);
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("</div>");
    }

    public void VisitServiceInstance (ServiceInstanceOptionInfo o)
    {
      // La partie "détail des méthodes" de doxygen se compose de trois parties :
      // - Un titre (h2 de classe .memtitle)
      // - Une partie "sous-titre" (div de classe .memitem.memproto)
      // - Une partie "description" (div de classe .memitem.memdoc)
      m_full_stream.WriteLine ("<h2 class=\"memtitle\" style=\"border-color: green;\">");
      _WriteColoredTitle ("green", o);
      //m_full_stream.WriteLine("<p>SERVICE TYPE={0}</p>",o.Type);
      m_full_stream.WriteLine ("</h2>");
      m_full_stream.WriteLine ("<div class=\"memitem\" style=\"border-color: green;\">");
      m_full_stream.WriteLine ("<div class=\"memproto\">");
      _AddBriefDescription (o, false);
      _AddFullDescription (o);
      CodeInterfaceInfo interface_info = null;
      if (m_code_info.Interfaces.TryGetValue (o.Type, out interface_info)) {
        if (interface_info.Services.Count>0) {
          //Console.WriteLine("SERVICE TYPE FOUND={0}",o.Type);
          m_full_stream.WriteLine ("<div class=\"ServiceTable\">");
          // TODO TRAD : Si traduction FR/EN : A traduire
          //m_full_stream.WriteLine ("<dl><dt>Valeur{0} possible{0} pour le tag <i>name</i>:</dt>",
          m_full_stream.WriteLine ("<dl><dt>Possible value{0} for tag <i>name</i>:</dt>",
                                   (interface_info.Services.Count>1)?"s":"");
          foreach (CodeServiceInfo csi in interface_info.Services) {
            //Console.WriteLine("SERVICE TYPE FOUND={0} {1} {2}",o.Type,csi.Name,csi.FileBaseName);
            if (csi.FileBaseName != null) {
              m_full_stream.WriteLine ("<dd>\\ref axldoc_service_{0} \"{1}\"<br/></dd>", csi.FileBaseName, csi.Name);
            } else {
              m_full_stream.WriteLine ("<dd>{0}</dd>", csi.Name);
            }
          }
          m_full_stream.WriteLine ("</dl>");
          m_full_stream.WriteLine ("</div>");
        }
      }
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("<div class=\"memdoc\">");
      _WriteCode(o);
      m_full_stream.WriteLine ("</div>");
      m_full_stream.WriteLine ("</div>");
    }

    private void _WriteDescription (int i, Option option, XmlElement desc_elem, TextWriter stream)
    {
      if (desc_elem != null) {
        foreach (XmlNode node in desc_elem) {
          if (node.NodeType == XmlNodeType.CDATA) {
            //Console.WriteLine ("** ** CDATA SECTION {0}", node.Value);

            // On est dans un contexte /htmlonly, donc il faut remplacer
            // les \n\n par des <br> (puisque doxygen ne le fera pas). 
            stream.Write (node.Value.Replace("\n\n", "<br>"));
          } 
          else {
            // NOTE GG: il faut utiliser node.OuterXml et pas (2) sinon les sous balises de la
            // description ne sont pas pris en compte.
            // Par exemple: <description>Test <b>très</b> important</description>.
            // Avec la méthode 2, cela donne: 'Test important' et la valeur entre des balises <b>
            // n'est pas prise en compte.
            stream.Write(node.OuterXml.Replace("\n\n", "<br>"));
            // (2) stream.Write (node.Value == null ? node.Value : node.Value.Trim ()); // Rk can be rewritten with VS 2015 as: node.Value ?.Trim()
          }
        }
        // Call extra description writer (working with XElement)
        IEnumerable<XElement> description_children = XElement.Parse (desc_elem.OuterXml).Descendants ();
        foreach (IExtraDescriptionWriter extra_description_writer in m_extra_description_writers) {
          extra_description_writer.writeDescription (option, description_children, stream);
        }
      } else {
        foreach (IExtraDescriptionWriter extra_description_writer in m_extra_description_writers) {
          extra_description_writer.writeDescription (option, null, stream);
        }
      }
    }

    private void _AddFullDescription (Option o)
    {
      _AddFullDescription (0, o, o.DescriptionElement);
    }

    private void _AddFullDescription (int i, Option option, XmlElement desc_elem)
    {
      // La description est récupérée brut, en html donc passage en \htmlonly
      m_full_stream.Write ("\\htmlonly <div class='OptionFullDescription'>");
      _WriteDescription (i, option, desc_elem, m_full_stream);
      m_full_stream.WriteLine ("</div> \\endhtmlonly");
    }

    private void _AddBriefDescription (Option o, bool use_subpage)
    {
      string href_name = DoxygenDocumentationUtils.AnchorName (o);
      string ref_type = use_subpage ? "subpage" : "ref";
      m_brief_stream.Write ("<li>");
      m_brief_stream.Write ("\\{2} {1} \"{0}\"", o.GetTranslatedName (m_code_info.Language), href_name, ref_type);
      // Si demandé, affiche les classes utilisateurs de cette option
      if (PrintUserClass) {
        string[] user_classes = o.UserClasses;
        if (user_classes != null && user_classes.Length > 0) {
          string v = "<span class='UserClassInBriefDesc'>[" + String.Join (",", user_classes) + "]</span>";
          m_brief_stream.Write (v);
        }
      }

      if (m_config.show_details) {
        if (o.MinOccurs != 1 || o.MaxOccurs != 1) {
          m_brief_stream.WriteLine (" [{0}...", o.MinOccurs);
          if (o.MaxOccurs != Option.UNBOUNDED)
            m_brief_stream.WriteLine ("{0}]", o.MaxOccurs);
          else
            m_brief_stream.WriteLine ("undefined]");
        }
        if (o.HasDefault)
          m_brief_stream.WriteLine (" has default value");
      }

      if (o is ComplexOptionInfo) {
        OptionTypeCounterVisitor otc = new OptionTypeCounterVisitor ();
        (o as ComplexOptionInfo).AcceptChildren (otc, x => m_config.private_app_pages.Filter(x));
        m_brief_stream.WriteLine (" ({0} options{1})", 
                                  otc.NbTotalOption,
                                  (use_subpage)?" in independant page":"");
      }
      m_brief_stream.WriteLine ("</li>");
    }

    private void _WriteColoredTitle (string color, Option o)
    {
      _WriteHtmlOnly (m_full_stream, "<font color = \"" + color + "\" >");
      _WriteTitle (o);
      _WriteHtmlOnly (m_full_stream, "</font>");
    }

    private void _WriteTitle (Option o)
    {
      string anchor_name = DoxygenDocumentationUtils.AnchorName (o);
      string translated_full_name = o.GetTranslatedFullName (m_code_info.Language);
      m_full_stream.WriteLine ("\\anchor {1} {0}",
                                    translated_full_name, anchor_name, o.GetType ().Name);
      if (m_dico_writer != null) {
        m_dico_writer.Write ("pagename=" + m_doc_file.PageName () + " frname=" + translated_full_name + " anchorname=" + anchor_name + "\n");
      }

      if (o.MinOccurs != 1 || o.MaxOccurs != 1) {
        m_full_stream.WriteLine ("[{0}...", o.MinOccurs);
        if (o.MaxOccurs != Option.UNBOUNDED)
          m_full_stream.WriteLine ("{0}]", o.MaxOccurs);
        else
          m_full_stream.WriteLine ("undefined]");
      }
      string type_name = o.Type;
      // Affiche le type de l'option et sa valeur par dÃ©faut si prÃ©sente
      if (type_name != null && !(o is ComplexOptionInfo)) {

        string unit_type = null;
        {
        SimpleOptionInfo simple_option = o as SimpleOptionInfo;
        if (simple_option != null && !String.IsNullOrWhiteSpace(simple_option.PhysicalUnit))
          unit_type = simple_option.PhysicalUnit;
        }

        string default_value = null;
        {
          if (o.HasDefault)
            default_value = o.DefaultValue;
        }

        {
          EnumerationOptionInfo enum_option = o as EnumerationOptionInfo;
          if (enum_option != null) {
            type_name = "enumeration";
            if (enum_option.HasDefault)
              default_value = enum_option.GetTranslatedEnumerationName (default_value, m_code_info.Language);
          }
        }

        m_full_stream.WriteLine (" ({0}{1}{2})", 
                                 type_name, 
                                 (default_value==null)?"":String.Format("={0}", default_value),
                                 (unit_type==null)?"":String.Format("; physical unit is {0}", unit_type));
      }
      // Si demandÃ©, affiche les classes utilisateurs de cette option
      if (PrintUserClass) {
        string[] user_classes = o.UserClasses;
        if (user_classes != null && user_classes.Length > 0) {
          string v = "<span class='UserClassInOptionTitle'>[" + String.Join (",", user_classes) + "]</span>";
          m_full_stream.WriteLine (v);
        }
      }
    }

    private void _WriteCode (Option o)
    {
      // Affiche un exemple de cette option (pour copier-coller)
      if (!(o is ComplexOptionInfo)) {
        string field_name = Utils.XmlGetAttributeValue (o.DescriptionElement, "field-name");
        if (field_name == null)
          field_name = "value";
        //m_full_stream.Write ("<div><pre class='OptionName'>");
        m_full_stream.WriteLine ("```xml");
        if (o is ServiceInstanceOptionInfo) {
          m_full_stream.Write ("<{0} name='{1}'>service configuration block</{0}>",
                                        o.GetTranslatedName (m_code_info.Language), field_name);
        } else
          m_full_stream.Write ("<{0}>{1}</{0}>",
                                        o.GetTranslatedName (m_code_info.Language), field_name);
        m_full_stream.WriteLine ("```");
      }
    }

    private void _WriteHRw30Tag (TextWriter stream)
    {
      _WriteHtmlOnly (stream, "<hr noshade size=2 width=30 align=left>\n");
    }

    private void _WriteHRTag (TextWriter stream)
    {
      _WriteHtmlOnly (stream, "<hr>\n");
    }

    private void _WriteHtmlOnly (TextWriter stream, string value)
    {
      stream.WriteLine ("\\htmlonly");
      stream.Write (value);
      stream.WriteLine ("\\endhtmlonly");
    }

    private void _WritePageFullDesc (Option option, XmlElement desc_elem)
    {
      TextWriter tw = m_doc_file.MainDescStream;
      tw.Write ("<div class='OptionFullDescription'>");
      _WriteDescription (0, option, desc_elem, tw);
      tw.WriteLine ("</div>");
    }

    private void _WriteOptionSeparator (Option o)
    {
      if (o.ParentOption != null)
        _WriteHRw30Tag (m_full_stream);
      else
        _WriteHRTag (m_full_stream);
    }
  }
}
