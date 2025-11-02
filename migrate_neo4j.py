#!/usr/bin/env python3
"""
Neo4j Migration Script: Local to AuraDB
Migrates all nodes, relationships, and indexes from local Neo4j to AuraDB
"""

import os
from neo4j import GraphDatabase
from typing import List, Dict, Any
import json
from tqdm import tqdm

class Neo4jMigrator:
    def __init__(self, source_uri: str, source_user: str, source_password: str,
                 target_uri: str, target_user: str, target_password: str):
        """
        Initialize the migrator with source and target database connections
        """
        self.source_driver = GraphDatabase.driver(source_uri, auth=(source_user, source_password))
        self.target_driver = GraphDatabase.driver(target_uri, auth=(target_user, target_password))
        
    def close(self):
        """Close database connections"""
        self.source_driver.close()
        self.target_driver.close()
    
    def export_nodes(self) -> List[Dict[str, Any]]:
        """Export all nodes from source database"""
        print("📤 Exporting nodes from local Neo4j...")
        
        with self.source_driver.session() as session:
            # Get all nodes with their labels and properties
            result = session.run("""
                MATCH (n)
                RETURN id(n) as node_id, labels(n) as labels, properties(n) as properties
            """)
            
            nodes = []
            for record in tqdm(result, desc="Exporting nodes"):
                nodes.append({
                    'id': record['node_id'],
                    'labels': record['labels'],
                    'properties': record['properties']
                })
            
            print(f"✅ Exported {len(nodes)} nodes")
            return nodes
    
    def export_relationships(self) -> List[Dict[str, Any]]:
        """Export all relationships from source database"""
        print("📤 Exporting relationships from local Neo4j...")
        
        with self.source_driver.session() as session:
            # Get all relationships with their types and properties
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN id(a) as start_id, id(b) as end_id, 
                       type(r) as rel_type, properties(r) as properties
            """)
            
            relationships = []
            for record in tqdm(result, desc="Exporting relationships"):
                relationships.append({
                    'start_id': record['start_id'],
                    'end_id': record['end_id'],
                    'type': record['rel_type'],
                    'properties': record['properties']
                })
            
            print(f"✅ Exported {len(relationships)} relationships")
            return relationships
    
    def import_nodes(self, nodes: List[Dict[str, Any]]):
        """Import nodes to target database"""
        print("📥 Importing nodes to AuraDB...")
        
        with self.target_driver.session() as session:
            # Clear existing data (optional - comment out if you want to keep existing data)
            print("🗑️  Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Import nodes in batches
            batch_size = 1000
            for i in tqdm(range(0, len(nodes), batch_size), desc="Importing nodes"):
                batch = nodes[i:i + batch_size]
                
                # Create nodes with dynamic labels and properties
                for node in batch:
                    labels_str = ':'.join(node['labels']) if node['labels'] else ''
                    if labels_str:
                        labels_str = ':' + labels_str
                    
                    # Create the node with properties
                    session.run(f"""
                        CREATE (n{labels_str})
                        SET n = $properties
                        SET n._original_id = $original_id
                    """, properties=node['properties'], original_id=node['id'])
        
        print(f"✅ Imported {len(nodes)} nodes")
    
    def import_relationships(self, relationships: List[Dict[str, Any]]):
        """Import relationships to target database"""
        print("📥 Importing relationships to AuraDB...")
        
        with self.target_driver.session() as session:
            # Import relationships in batches
            batch_size = 1000
            for i in tqdm(range(0, len(relationships), batch_size), desc="Importing relationships"):
                batch = relationships[i:i + batch_size]
                
                for rel in batch:
                    # Find nodes by their original IDs and create relationship
                    session.run(f"""
                        MATCH (a {{_original_id: $start_id}})
                        MATCH (b {{_original_id: $end_id}})
                        CREATE (a)-[r:{rel['type']}]->(b)
                        SET r = $properties
                    """, start_id=rel['start_id'], end_id=rel['end_id'], 
                        properties=rel['properties'])
        
        print(f"✅ Imported {len(relationships)} relationships")
    
    def cleanup_temp_properties(self):
        """Remove temporary _original_id properties"""
        print("🧹 Cleaning up temporary properties...")
        
        with self.target_driver.session() as session:
            session.run("MATCH (n) REMOVE n._original_id")
        
        print("✅ Cleanup completed")
    
    def export_indexes_and_constraints(self):
        """Export indexes and constraints"""
        print("📤 Exporting indexes and constraints...")
        
        with self.source_driver.session() as session:
            # Get indexes
            indexes = session.run("SHOW INDEXES").data()
            constraints = session.run("SHOW CONSTRAINTS").data()
            
            print(f"Found {len(indexes)} indexes and {len(constraints)} constraints")
            return indexes, constraints
    
    def migrate_all(self):
        """Perform complete migration"""
        try:
            print("🚀 Starting Neo4j migration from local to AuraDB...")
            
            # Test connections
            print("🔗 Testing connections...")
            with self.source_driver.session() as session:
                session.run("RETURN 1")
            print("✅ Source connection OK")
            
            with self.target_driver.session() as session:
                session.run("RETURN 1")
            print("✅ Target connection OK")
            
            # Export data
            nodes = self.export_nodes()
            relationships = self.export_relationships()
            
            # Save backup (optional)
            print("💾 Saving backup files...")
            with open('nodes_backup.json', 'w') as f:
                json.dump(nodes, f, indent=2)
            with open('relationships_backup.json', 'w') as f:
                json.dump(relationships, f, indent=2)
            print("✅ Backup files saved")
            
            # Import data
            self.import_nodes(nodes)
            self.import_relationships(relationships)
            
            # Cleanup
            self.cleanup_temp_properties()
            
            # Export and display indexes/constraints info
            indexes, constraints = self.export_indexes_and_constraints()
            if indexes or constraints:
                print("\n⚠️  Manual step required:")
                print("You'll need to recreate these indexes and constraints manually in AuraDB:")
                for idx in indexes:
                    print(f"  - Index: {idx}")
                for const in constraints:
                    print(f"  - Constraint: {const}")
            
            print("\n🎉 Migration completed successfully!")
            print(f"📊 Migrated {len(nodes)} nodes and {len(relationships)} relationships")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            raise
        finally:
            self.close()

def main():
    """Main migration function"""
    print("Neo4j Migration Tool: Local → AuraDB")
    print("=" * 50)
    
    # Source (Local Neo4j) configuration
    SOURCE_URI = "bolt://localhost:7687"
    SOURCE_USER = "neo4j"
    SOURCE_PASSWORD = input("Enter your LOCAL Neo4j password: ")
    
    print("\nNow enter your AuraDB credentials:")
    # Target (AuraDB) configuration
    TARGET_URI = input("Enter AuraDB URI (neo4j+s://xxxxx.databases.neo4j.io): ")
    TARGET_USER = input("Enter AuraDB username (usually 'neo4j'): ")
    TARGET_PASSWORD = input("Enter AuraDB password: ")
    
    # Confirm migration
    print(f"\n📋 Migration Summary:")
    print(f"  Source: {SOURCE_URI}")
    print(f"  Target: {TARGET_URI}")
    
    confirm = input("\n⚠️  This will REPLACE all data in AuraDB. Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Migration cancelled.")
        return
    
    # Perform migration
    migrator = Neo4jMigrator(
        SOURCE_URI, SOURCE_USER, SOURCE_PASSWORD,
        TARGET_URI, TARGET_USER, TARGET_PASSWORD
    )
    
    migrator.migrate_all()

if __name__ == "__main__":
    main()
