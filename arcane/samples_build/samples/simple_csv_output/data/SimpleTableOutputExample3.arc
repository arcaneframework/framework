<?xml version="1.0"?>
<case codename="csv" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Examples CSV</title>
    <timeloop>example3</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <face-numbering-version>1</face-numbering-version>

        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>

        <origin>0.0 0.0</origin>

        <x>
          <length>1.0</length>
          <n>1</n>
        </x>

        <y>
          <length>1.0</length>
          <n>1</n>
        </y>

      </generator>
    </mesh>
  </meshes>

  <simple-table-output-example3>
    <csv-output name="SimpleCsvOutput">
      <!-- Les noms sont utilisés directement par le service CSV. -->
      <!-- Le nom du répertoire à créer/utiliser. -->
      <tableDir>example3</tableDir>
      <!-- Le nom du fichier à créer/écraser. -->
      <tableName>Results_Example3</tableName>

      <!-- Au final, on aura un fichier ayant comme chemin : ./example3/Results_Example3.csv -->
    </csv-output>
  </simple-table-output-example3>

</case>
