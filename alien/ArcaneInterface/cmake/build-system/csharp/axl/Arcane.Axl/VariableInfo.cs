using System.Xml;
using Integer = System.Int32;
using System;

namespace Arcane.Axl
{
  /*!
   * \brief Informations de déclaration d'une variable dans le fichier axl.
   */
  public class VariableInfo : XmlInfo
  {
    /**
      * Valeur de l'attribut XML "field-name", nom de l'attribut représentant
      * la variable dans la classe générée.
      */
    string m_field_name;
    public string FieldName { get { return m_field_name; } }

    string m_name;
    //! Nom de la variable
    public string Name { get { return m_name; } }

    Integer m_dim;
    /** Valeur de l'attribut XML "dim", dimension de la variable: 0, 1 ou 2. */
    public Integer Dimension { get { return m_dim; } }

    ItemKind m_item_kind;
    /**
     * Valeur de l'attribut XML "item-kind",
     * type des éléments du maillage sur lequel repose la variable.
     */
    public ItemKind ItemKind { get { return m_item_kind; } }

    string m_data_type;
    /** Valeur de l'attribut XML "data-type", type des éléments de la variable. */
    public string DataType { get { return m_data_type; } }

    bool m_no_dump;
    /**
     * Valeur de l'attribut XML "dump",
     * indiquant si la variable doit etre sauvegardée.
     */
    public bool IsNoDump { get { return m_no_dump; } }

    bool m_no_need_sync;
    /**
     * Valeur de l'attribut XML "need-sync",
     * indiquant si la variable doit etre synchronisée.
     */
    public bool IsNoNeedSync { get { return m_no_need_sync; } }

    bool m_execution_depend;
    /**
     * Valeur de l'attribut XML "execution-depend",
     * indiquant si la valeur de la variable dépend de l'exécution (par exemple le temps CPU).
     */
    public bool IsExecutionDepend { get { return m_execution_depend; } }

    bool m_sub_domain_depend;
    /**
     * Valeur de l'attribut XML "sub-domain-depend",
     * indiquant si la valeur de la variable dépend du découpage.
     */
    public bool IsSubDomainDepend { get { return m_sub_domain_depend; } }

    bool m_sub_domain_private;
    /**
     * Valeur de l'attribut XML "sub-domain-private",
     * indiquant si la valeur de la variable dépend du sous-domaine.
     */
    public bool IsSubDomainPrivate { get { return m_sub_domain_private; } }

    bool m_no_restore;
    /**
     * Valeur de l'attribut XML "no-restore",
     * indiquant si la valeur de la variable doit être restaurée en cas de retour-arrière.
     */
    public bool IsNoRestore { get { return m_no_restore; } }

    public VariableInfo(XmlElement node)
    {
      m_dim = 0;
      m_field_name = node.GetAttribute("field-name");
      if (m_field_name == null)
        AttrError(node, "field-name");
      m_name = node.GetAttribute("name");
      if (m_name == null)
        AttrError(node, "name");

      // dimension
      bool has_error = false;
      string dim = node.GetAttribute("dim");
      if (dim == "0" || dim == "1" || dim == "2") {
        m_dim = int.Parse(dim);
      }
      else
        has_error = true;
      if (has_error) {
        Console.WriteLine("** ERREUR: attribut \"dim\" de l'option <{0}> invalide ({1})\n",
                          node.Name, dim);
        Console.WriteLine("** Les dimensions reconnues sont 0, 1 et 2.");
        Error(node, "invalid dimension");
      }

      // kind
      string item_kind = node.GetAttribute("item-kind");
      if (item_kind == "node")
        m_item_kind = ItemKind.Node;
      else if (item_kind == "edge")
        m_item_kind = ItemKind.Edge;
      else if (item_kind == "face")
        m_item_kind = ItemKind.Face;
      else if (item_kind == "cell")
        m_item_kind = ItemKind.Cell;
      else if (item_kind == "dualnode")
        m_item_kind = ItemKind.DualNode;
      else if (item_kind == "link")
        m_item_kind = ItemKind.Link;
      else if (item_kind == "none")
        m_item_kind = ItemKind.Unknown;
      else {
        Console.WriteLine("** ERREUR: attribut \"item-kind\" de l'option <" + node.Name
             + "> invalide (" + item_kind + ").\n");
        Console.WriteLine("** Les types reconnus sont 'node', 'edge', "
             + "'face', 'cell', 'link', 'dualnode' et 'none'.\n");
        Error(node, "mauvaise valeur pour l'attribut \"kind\"");
      }

      // type
      string data_type = node.GetAttribute("data-type");
      if (data_type == "bool")
        m_data_type = "Byte";
      else if (data_type == "byte")
        m_data_type = "Byte";
      else if (data_type == "real")
        m_data_type = "Real";
      else if (data_type == "integer")
        m_data_type = "Integer";
      else if (data_type == "int64")
        m_data_type = "Int64";
      else if (data_type == "string")
        m_data_type = "String";
      else if (data_type == "real2")
        m_data_type = "Real2";
      else if (data_type == "real2x2")
        m_data_type = "Real2x2";
      else if (data_type == "real3")
        m_data_type = "Real3";
      else if (data_type == "real3x3")
        m_data_type = "Real3x3";
      else {
        Console.WriteLine("** ERREUR: attribut \"data-type\" de l'option <" + node.Name
            + "> invalide (" + data_type + ").\n");
        Console.WriteLine("** Les types reconnus sont 'bool', 'byte', 'real', 'integer', "
             + "'int64', 'string', 'real2', 'real2x2' "
             + "'real3' et 'real3x3'.\n");
        Error(node, "mauvaise valeur pour l'attribut \"data-type\"");
      }

      m_no_dump = !_ReadProperty(node,"dump");
      m_no_need_sync = !_ReadProperty(node,"need-sync");
      m_execution_depend = _ReadProperty(node,"execution-depend");
      m_sub_domain_private = _ReadProperty(node,"sub-domain-private");
      m_no_restore = _ReadProperty(node,"no-restore");
      if (GlobalContext.Instance.NoRestore == true){
         m_no_restore = true;
      }
      m_sub_domain_depend = _ReadProperty(node,"sub-domain-depend");
    }

    bool _ReadProperty(XmlElement node,string name)
    {
      string v = node.GetAttribute(name);
      return (v == "true" || v == "1");
    }

    /**
     * \brief Nom de la classe et du namespace pour cette variable.
     * 
     * Par exemple, pour une variable aux mailles scalaires, le nom
     * est 'VariableCellReal' et le namespace 'Arcane'.
     */
    public virtual void GetGeneratedClassName(out string class_name,out string namespace_name)
    {
      string name = "Variable";
      if (m_item_kind != ItemKind.Unknown)
        name += m_item_kind;
      switch (m_dim) {
      case 0:
        if (m_item_kind == ItemKind.Unknown)
          name+= "Scalar";
        break;
      case 1:
        name += "Array";
        break;
      case 2:
        name += "Array2";
        break;
      }
      name += m_data_type;

      class_name = name;
      namespace_name = "Arcane";
    }

  }
}
