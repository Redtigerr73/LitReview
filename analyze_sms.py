"""
SMS Analysis Runner with SRE Taxonomy
Integrates systematic mapping study analysis with specialized SRE taxonomy
"""

import pandas as pd
from src.sms_analyzer import SMSAnalyzer
from src.taxonomy.taxonomy_integrator import TaxonomyIntegrator
from src.utils.logger import setup_logger
import logging
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Run Systematic Mapping Study analysis with SRE taxonomy')
    parser.add_argument('--input', default='data/articles.xlsx',
                       help='Input file with collected articles')
    parser.add_argument('--output', default='data/sms_analysis',
                       help='Output directory for analysis results')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('sms_analysis')

    try:
        # Verify input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return

        # Load existing data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_excel(args.input)
        logger.info(f"Loaded {len(df)} articles")

        # 1. Process taxonomy
        logger.info("Applying SRE taxonomy classification...")
        taxonomy_integrator = TaxonomyIntegrator()
        df = taxonomy_integrator.process_papers(df)
        
        # 2. Generate taxonomy visualizations
        logger.info("Generating taxonomy visualizations...")
        taxonomy_integrator.generate_visualizations(df, f"{args.output}/taxonomy")

        # 3. Run standard SMS analysis
        logger.info("Running general SMS analysis...")
        analyzer = SMSAnalyzer()
        
        logger.info("Analyzing temporal trends...")
        trends = analyzer.analyze_temporal_trends(df)

        logger.info("Classifying research...")
        df = analyzer.classify_research(df)

        logger.info("Generating knowledge graph...")
        knowledge_graph = analyzer.generate_knowledge_graph(df)

        # Export results
        logger.info(f"Exporting analysis results to {args.output}")
        analyzer.export_analysis(df, args.output)

        logger.info("Analysis complete! Check the following files:")
        logger.info(f"1. Taxonomy visualizations: {args.output}/taxonomy/")
        logger.info(f"2. General SMS analysis: {args.output}/")
        logger.info(f"3. Taxonomy statistics: data/taxonomy/taxonomy_stats.json")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
