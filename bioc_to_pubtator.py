#!/usr/bin/env python3
"""
Convert BioC XML to PubTator format.
- First column: 8-digit PMID (zero-padded if needed)
- Relations with 'relatedTo' become 'Association'
- Output format matches PubTator standard
"""

import re
from xml.etree import ElementTree as ET

def parse_bioc_to_pubtator(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    output_lines = []
    
    for document in root.findall('.//document'):
        doc_id_raw = document.find('id').text
        
        numeric_match = re.match(r'(\d+)', doc_id_raw)
        if numeric_match:
            pmid = numeric_match.group(1).zfill(8)
        else:
            print(f"Warning: Could not extract numeric ID from {doc_id_raw}, skipping")
            continue
        
        for passage in document.findall('.//passage'):
            text_elem = passage.find('text')
            full_text = text_elem.text if text_elem is not None else ""
            
            # Remove "** IGNORE LINE **" parts
            clean_text = re.sub(r'\*\* IGNORE LINE \*\*\n?', '', full_text).strip()
            
            # Split into title (first line) and abstract (rest)
            lines = clean_text.split('\n', 1)
            title = lines[0].strip() if lines else ""
            abstract = lines[1].strip() if len(lines) > 1 else ""
            
            output_lines.append(f"{pmid}|t|{title}")
            output_lines.append(f"{pmid}|a|{abstract}")
            
            # Build annotation lookup
            annotations = {}
            for anno in passage.findall('annotation'):
                anno_id = anno.get('id')
                anno_type = None
                
                for infon in anno.findall('infon'):
                    if infon.get('key') == 'type':
                        anno_type = infon.text
                        break
                
                location = anno.find('location')
                offset = int(location.get('offset', 0)) if location is not None else 0
                length = int(location.get('length', 0)) if location is not None else 0
                
                text_node = anno.find('text')
                anno_text = text_node.text if text_node is not None else ""
                
                annotations[anno_id] = {
                    'id': anno_id, 
                    'type': anno_type, 
                    'offset': offset, 
                    'length': length, 
                    'text': anno_text
                }
                
                end_offset = offset + length
                output_lines.append(f"{pmid}\t{offset}\t{end_offset}\t{anno_text}\t{anno_type}\t{anno_id}")
            
            # Relations
            for relation in passage.findall('relation'):
                rel_type = None
                for infon in relation.findall('infon'):
                    if infon.get('key') == 'relation type':
                        rel_type = infon.text
                        break
                
                if rel_type == 'relatedTo':
                    rel_type = 'Association'
                
                nodes = relation.findall('node')
                if len(nodes) >= 2:
                    arg1_id = nodes[0].get('refid')
                    arg2_id = nodes[1].get('refid')
                    
                    output_lines.append(f"{pmid}\t{rel_type}\t{arg1_id}\t{arg2_id}")
            
            output_lines.append("")  # Blank line between documents
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Converted to {output_file}")
    print(f"Total lines: {len(output_lines)}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python bioc_to_pubtator.py input.xml output.pubtator")
        print("   Or import and call: parse_bioc_to_pubtator('input.xml', 'output.pubtator')")
        sys.exit(1)
    
    parse_bioc_to_pubtator(sys.argv[1], sys.argv[2])
