/*---------------------------------------------------------------------------*/
/* DoxygenDocumentationVisitor.cs                              (C) 2000-2012 */
/*                                                                           */
/* GÃ©nÃ©ration de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace Arcane.Axl
{
    public class DoxygenDocumentationVisitor
     : IOptionInfoVisitor
    {
        private TextWriter m_brief_stream;
        private TextWriter m_full_stream;
        private DoxygenDocumentationFile m_doc_file;
        private CodeInfo m_code_info;
        private bool m_print_user_class;
        private StreamWriter m_dico_writer = null;

        public bool PrintUserClass { get { return m_print_user_class; } set { m_print_user_class = value; } }

        public DoxygenDocumentationVisitor(DoxygenDocumentationFile doc_file, CodeInfo code_info)
        {
            m_full_stream = doc_file.FullDescStream;
            m_brief_stream = doc_file.BriefDescStream;
            m_doc_file = doc_file;
            m_code_info = code_info;
        }

        public DoxygenDocumentationVisitor(DoxygenDocumentationFile doc_file, CodeInfo code_info, StreamWriter dico_writer)
        {
            m_full_stream = doc_file.FullDescStream;
            m_brief_stream = doc_file.BriefDescStream;
            m_doc_file = doc_file;
            m_code_info = code_info;
            m_dico_writer = dico_writer;
        }

        public void VisitServiceOrModule(ServiceOrModuleInfo info)
        {
            ServiceInfo sinfo = info as ServiceInfo;
            if (sinfo != null)
            {
                StringBuilder sb = new StringBuilder();
                foreach (ServiceInfo.Interface ii in sinfo.Interfaces)
                {
                    sb.AppendFormat("(implements '#{0}' interface)", ii.Name);
                }
                m_doc_file.SubTitle = sb.ToString();
            }


            _WritePageFullDesc(info.DescriptionElement);
            // Pour retrouver facilement une option dans la doc, on la trie
            // pas ordre alphabetique
            SortedList<string, Option> sorted_options = new SortedList<string, Option>();
            foreach (Option o in info.Options)
            {
                string name = o.GetTranslatedName(m_code_info.Language);
                // Attention, il est possible par erreur que deux options aient le mÃªme nom traduit.
                // Dans cas, le Add va envoyer une exception. Pour que l'erreur soit plus facile
                // Ã  trouver, on envoie un message d'erreur.
                if (sorted_options.ContainsKey(name))
                {
                    throw new ApplicationException(String.Format("L'option {0} a la mÃªme la traduction '{1}' qu'une autre option de mÃªme niveau hiÃ©rarchique",
                                                                 o.GetIdString(), name));
                }
                sorted_options.Add(name, o);
            }
            foreach (KeyValuePair<string, Option> opt in sorted_options)
            {
                opt.Value.Accept(this);
            }
        }

        public void VisitComplex(ComplexOptionInfo o)
        {
            OptionTypeCounterVisitor otc = new OptionTypeCounterVisitor();
            o.AcceptChildren(otc);

            int max_display = 30;
            if (max_display != (-1) && otc.NbTotalOption > max_display)
            {
                _AddBriefDescription(o, true);
                m_brief_stream.WriteLine("({0} options)", otc.NbTotalOption);
                DoxygenDocumentationFile df = new DoxygenDocumentationFile(o, m_doc_file.OutputPath, m_code_info.Language);
                DoxygenDocumentationVisitor dv = null;
                if (m_dico_writer != null)
                {
                    dv = new DoxygenDocumentationVisitor(df, m_code_info, m_dico_writer);
                }
                else
                {
                    dv = new DoxygenDocumentationVisitor(df, m_code_info);
                }
                dv._WritePageFullDesc(o.DescriptionElement);
                o.AcceptChildrenSorted(dv, "fr");
                df.Write();
            }
            else
            {
                _WriteHRw30Tag(m_full_stream);
                _AddBriefDescription(o, false);
                m_brief_stream.WriteLine("({0} options)", otc.NbTotalOption);
                _WriteHtmlOnly(m_full_stream, "<div class=\"ComplexOptionInfoBlock\">");
                _WriteColoredTitle("purple", o);
                _AddFullDescription(o);
                if (otc.NbTotalOption > 0)
                {
                    m_brief_stream.WriteLine("<ul>");
                    o.AcceptChildrenSorted(this, "fr");
                    m_brief_stream.WriteLine("</ul>");
                }
                _WriteHtmlOnly(m_full_stream, "</div><!-- End of ComplexOptionInfoBlock -->"); // Pour ComplexOptionInfoBlock
            }
        }

        public void VisitEnumeration(EnumerationOptionInfo o)
        {
            _WriteHRw30Tag(m_full_stream);
            _WriteColoredTitle("olive", o);
            XmlDocument owner_doc = o.Node.OwnerDocument;
            XmlElement desc_elem = o.DescriptionElement;
            // Construit dans \a enum_list_element
            // le tableau contenant les descriptions des diffÃ©rentes valeurs
            // que peut prendre l'Ã©numÃ©ration.
            // Ce tableau est ensuite insÃ©rer dans la description, Ã  l'endroit
            // oÃ¹ se trouve l'Ã©lÃ©ment <enum-description> ou s'il est absent,
            // aprÃ¨s la description brÃ¨ve.
            XmlElement enum_list_element = owner_doc.CreateElement("div");
            enum_list_element.SetAttribute("class", "EnumTable");
            enum_list_element.AppendChild(owner_doc.CreateElement("br"));
            XmlElement table_element = owner_doc.CreateElement("table");
            enum_list_element.AppendChild(table_element);
            {
                // Affiche les titres
                XmlElement tr = owner_doc.CreateElement("tr");
                table_element.AppendChild(tr);
                XmlElement th = owner_doc.CreateElement("th");
                tr.AppendChild(th);
                string field_name = Utils.XmlGetAttributeValue(desc_elem, "field-name");
                if (field_name == null)
                    field_name = "value";
                th.InnerText = field_name;
                th = owner_doc.CreateElement("th");
                tr.AppendChild(th);
                th.InnerText = "description";
            }
            foreach (EnumValueOptionInfo ev in o.EnumValues)
            {
                XmlElement tr = owner_doc.CreateElement("tr");
                table_element.AppendChild(tr);
                //sw.Write("<tr>");
                XmlElement td1 = owner_doc.CreateElement("td");
                tr.AppendChild(td1);

                td1.InnerText = ev.GetTranslatedName(m_code_info.Language);

                XmlElement td2 = owner_doc.CreateElement("td");
                XmlElement sub_desc_elem = ev.DescriptionElement;
                if (sub_desc_elem != null)
                {
                    //XmlElement sub_desc_clone = (XmlElement)sub_desc_elem.CloneNode(true);
                    XmlNodeList children = sub_desc_elem.ChildNodes;
                    foreach (XmlNode child in children)
                    {
                        td2.AppendChild(child.CloneNode(true));
                    }
                }
                tr.AppendChild(td2);
            }
            enum_list_element.AppendChild(owner_doc.CreateElement("br"));
            _AddBriefDescription(o, false);
            if (desc_elem != null)
            {
                XmlElement elem_description = desc_elem.SelectSingleNode("enum-description") as XmlElement;
                if (elem_description != null)
                {
                    Console.WriteLine("HAS ELEMENT DESCRIPTION {0} {1}", o.Name, enum_list_element.OuterXml);
                    //elem_description.InnerText = sw.ToString();
                    XmlNode desc_parent = elem_description.ParentNode;
                    desc_parent.ReplaceChild(enum_list_element, elem_description);
                }
                else
                    desc_elem.PrependChild(enum_list_element); //m_full_stream.Write(enum_list_element.OuterXml);
            }
            _AddFullDescription(o);
        }
        public void VisitExtended(ExtendedOptionInfo o)
        {
            _WriteHRw30Tag(m_full_stream);
            _WriteColoredTitle("teal", o);
            _AddBriefDescription(o, false);
            _AddFullDescription(o);
        }
        public void VisitScript(ScriptOptionInfo o)
        {
            _WriteHRw30Tag(m_full_stream);
            _WriteColoredTitle("teal", o);
            _AddBriefDescription(o, false);
            _AddFullDescription(o);
        }
        public void VisitSimple(SimpleOptionInfo o)
        {
            _WriteHRw30Tag(m_full_stream);
            _WriteColoredTitle("green", o);
            //m_full_stream.WriteLine("<p>TYPE={0}</p>",o.Type);
            //if (o.DefaultValue!=null)
            //m_full_stream.WriteLine("<p>DEFAULT={0}</p>",o.DefaultValue);
            _AddBriefDescription(o, false);
            _AddFullDescription(o);
        }
        public void VisitServiceInstance(ServiceInstanceOptionInfo o)
        {
            _WriteHRw30Tag(m_full_stream);
            _WriteColoredTitle("green", o);
            //m_full_stream.WriteLine("<p>SERVICE TYPE={0}</p>",o.Type);
            _AddBriefDescription(o, false);
            CodeInterfaceInfo interface_info = null;
            if (m_code_info.Interfaces.TryGetValue(o.Type, out interface_info))
            {
                //Console.WriteLine("SERVICE TYPE FOUND={0}",o.Type);
                m_full_stream.WriteLine("<div class=\"ServiceTable\"><br>");
                m_full_stream.WriteLine("<table>");
                foreach (CodeServiceInfo csi in interface_info.Services)
                {
                    //Console.WriteLine("SERVICE TYPE FOUND={0} {1} {2}",o.Type,csi.Name,csi.FileBaseName);
                    if (csi.FileBaseName != null)
                    {
                        m_full_stream.WriteLine("<tr><td>\\ref axldoc_service_{0} \"{1}\"</td></tr>", csi.FileBaseName, csi.Name);
                    }
                    else
                    {
                        m_full_stream.WriteLine("<tr><td>{0}</td></tr>", csi.Name);
                    }
                }
                m_full_stream.WriteLine("</table>");
                m_full_stream.WriteLine("<br></div>");
            }
            _AddFullDescription(o);
        }

        private void _WriteDescription(XmlElement desc_elem, TextWriter stream)
        {
            foreach (XmlNode node in desc_elem)
            {
                if (node.NodeType == XmlNodeType.CDATA)
                {
                    Console.WriteLine("** ** CDATA SECTION {0}", node.Value);
                    stream.Write(node.Value);
                }
                else
                    stream.Write(node.OuterXml);
            }
        }

        private void _AddFullDescription(Option o)
        {
            _AddFullDescription(o.DescriptionElement);
        }

        private void _AddFullDescription(XmlElement desc_elem)
        {
            if (desc_elem != null)
            {
                m_full_stream.Write("<div class='OptionFullDescription'>");
                _WriteDescription(desc_elem, m_full_stream);
                m_full_stream.WriteLine("</div>");
            }
        }

        private void _AddBriefDescription(Option o, bool use_subpage)
        {
            string href_name = DoxygenDocumentationUtils.AnchorName(o);
            string ref_type = use_subpage ? "subpage" : "ref";
            m_brief_stream.Write("<li>");
            m_brief_stream.Write("\\{2} {1} \"{0}\"", o.GetTranslatedName(m_code_info.Language), href_name, ref_type);
            // Si demandÃ©, affiche les classes utilisateurs de cette option
            if (PrintUserClass)
            {
                string[] user_classes = o.UserClasses;
                if (user_classes != null && user_classes.Length > 0)
                {
                    string v = "<span class='UserClassInBriefDesc'>[" + String.Join(",", user_classes) + "]</span>";
                    m_brief_stream.Write(v);
                }
            }
            m_brief_stream.WriteLine("</li>");
        }

        private void _WriteColoredTitle(string color, Option o)
        {
            _WriteHtmlOnly(m_full_stream, "<font color = \"" + color + "\" >");
            _WriteTitle(o);
            _WriteHtmlOnly(m_full_stream, "</font>");
        }

        private void _WriteTitle(Option o)
        {
            string anchor_name = DoxygenDocumentationUtils.AnchorName(o);
            string translated_full_name = o.GetTranslatedFullName(m_code_info.Language);
            m_full_stream.WriteLine("\\anchor {1}\n<div class='{2}'><strong>{0}",
                                    translated_full_name, anchor_name, o.GetType().Name);
            if (m_dico_writer != null)
            {
                m_dico_writer.Write("pagename=" + m_doc_file.PageName() + " frname=" + translated_full_name + " anchorname=" + anchor_name + "\n");
            }

            if (o.MinOccurs != 1 || o.MaxOccurs != 1)
            {
                m_full_stream.WriteLine("[{0}...", o.MinOccurs);
                if (o.MaxOccurs != Option.UNBOUNDED)
                    m_full_stream.WriteLine("{0}]", o.MaxOccurs);
                else
                    m_full_stream.WriteLine("undefined]");
            }
            m_full_stream.WriteLine("</strong>");
            string type_name = o.Type;
            // Affiche le type de l'option et sa valeur par dÃ©faut si prÃ©sente
            if (type_name != null && !(o is ComplexOptionInfo))
            {
                EnumerationOptionInfo enum_option = o as EnumerationOptionInfo;
                if (enum_option != null)
                    type_name = "enumeration";
                if (o.HasDefault)
                {
                    string default_value = o.DefaultValue;
                    if (enum_option != null)
                        default_value = enum_option.GetTranslatedEnumerationName(default_value, m_code_info.Language);
                    m_full_stream.WriteLine(" ({0}={1}) ", type_name, default_value);
                }
                else
                    m_full_stream.WriteLine(" ({0}) ", type_name);
            }
            // Si demandÃ©, affiche les classes utilisateurs de cette option
            if (PrintUserClass)
            {
                string[] user_classes = o.UserClasses;
                if (user_classes != null && user_classes.Length > 0)
                {
                    string v = "<span class='UserClassInOptionTitle'>[" + String.Join(",", user_classes) + "]</span>";
                    m_full_stream.WriteLine(v);
                }
            }
            m_full_stream.WriteLine("</div>\n");
            // Affiche un exemple de cette option (pour copier-coller)
            if (!(o is ComplexOptionInfo))
            {
                string field_name = Utils.XmlGetAttributeValue(o.DescriptionElement, "field-name");
                if (field_name == null)
                    field_name = "value";
                m_full_stream.Write("<div><pre class='OptionName'>");
                if (o is ServiceInstanceOptionInfo)
                {
                    m_full_stream.Write("&lt;{0} name='{1}'&gt; &lt;/{0}&gt;",
                                        o.GetTranslatedName(m_code_info.Language), field_name);
                }
                else
                    m_full_stream.Write("&lt;{0}&gt;{1}&lt;/{0}&gt;",
                                        o.GetTranslatedName(m_code_info.Language), field_name);
                m_full_stream.WriteLine("</pre></div>");
            }
        }

        private void _WriteHRw30Tag(TextWriter stream)
        {
            _WriteHtmlOnly(stream, "<hr noshade size=2 width=30 align=left />\n");
        }

        private void _WriteHRTag(TextWriter stream)
        {
            _WriteHtmlOnly(stream, "<hr />\n");
        }

        private void _WriteHtmlOnly(TextWriter stream, string value)
        {
            stream.WriteLine("\\htmlonly");
            stream.Write(value);
            stream.WriteLine("\\endhtmlonly");
        }

        private void _WritePageFullDesc(XmlElement desc_elem)
        {
            _WriteHRTag(m_full_stream);
            //_AddFullDescription(desc_elem);
            if (desc_elem != null)
            {
                TextWriter tw = m_doc_file.MainDescStream;
                tw.Write("<hr /><div class='OptionFullDescription'>");
                _WriteDescription(desc_elem, tw);
                tw.WriteLine("</div><hr />");
            }
            //_WriteHRTag(m_full_stream);
            m_full_stream.WriteLine("<h2 class='case_fulldesc'>Detailed list of options</h2>");
            //_WriteHRTag(m_full_stream);
        }
    }
}