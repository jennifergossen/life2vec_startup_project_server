"""
Startup Data Cleaning Module

This module provides comprehensive data cleaning functionality for startup datasets,
specifically designed to prepare data for the life2vec-style transformer models.

"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from collections import Counter
import pickle
from typing import Tuple, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StartupDataCleaner:
    """
    Comprehensive data cleaner for startup datasets.
    
    This class handles all aspects of cleaning startup data including:
    - Date filtering and validation
    - Country code filtering
    - Comma-separated field parsing
    - Education degree and subject standardization
    - Job title normalization with top-N filtering
    - Continuous variable binning
    - Data quality validation
    """
    
    def __init__(self, output_dir: str = "data/cleaned/cleaned_startup"):
        """
        Initialize the cleaner.
        
        Args:
            output_dir: Directory to save cleaned data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Standardization mappings
        self.degree_patterns = {
            'BACHELOR': r'BACHELOR|B\.A\.?|B\.S\.?|B\.SC\.?|B\.TECH|BTECH|B\.E\.?|B\.ENG|B\.COM|BBA|B\.B\.A|BSBA|BSE|BASC|AB\b|UNDERGRADUATE',
            'MASTER': r'MASTER|M\.A\.?|M\.S\.?|M\.SC\.?|MBA|M\.B\.A|M\.TECH|MTECH|M\.E\.?|M\.ENG|M\.COM|EXECUTIVE MBA|GRADUATE',
            'PHD': r'PHD|PH\.D\.?|DOCTORATE|DOCTORAL|DOCTOR OF PHILOSOPHY|D\.PHIL',
            'ASSOCIATE': r'ASSOCIATE|A\.A\.?|A\.S\.?|AA\b|AS\b',
            'CERTIFICATE': r'CERTIFICATE|CERT\.?|DIPLOMA|EXECUTIVE EDUCATION|CERTIFICATION',
            'JD': r'JURIS DOCTOR|J\.D\.?|JD\b|LLB|LL\.B|LAW DEGREE',
            'MD': r'DOCTOR OF MEDICINE|M\.D\.?|MD\b|MEDICINE|MEDICAL',
            'HIGH_SCHOOL': r'HIGH SCHOOL|SECONDARY|MATRICULATION|DIPLOMA'
        }
        
        self.job_title_patterns = {
            r'CHIEF EXECUTIVE OFFICER|CEO\b': 'CEO',
            r'CHIEF TECHNOLOGY OFFICER|CHIEF TECHNICAL OFFICER|CTO\b': 'CTO',
            r'CHIEF FINANCIAL OFFICER|CFO\b': 'CFO',
            r'CHIEF OPERATING OFFICER|COO\b': 'COO',
            r'CHIEF MARKETING OFFICER|CMO\b': 'CMO',
            r'CHIEF PRODUCT OFFICER|CPO\b': 'CPO',
            r'CHIEF REVENUE OFFICER|CRO\b': 'CRO',
            r'CHIEF DATA OFFICER|CDO\b': 'CDO',
            r'CO-FOUNDER|COFOUNDER|CO FOUNDER': 'CO-FOUNDER',
            r'\bFOUNDER\b': 'FOUNDER',
            r'\bPRESIDENT\b': 'PRESIDENT',
            r'VICE PRESIDENT|VP\b': 'VP',
            r'MANAGING DIRECTOR': 'MANAGING_DIRECTOR',
            r'EXECUTIVE DIRECTOR': 'EXECUTIVE_DIRECTOR',
            r'\bDIRECTOR\b': 'DIRECTOR',
            r'\bMANAGER\b': 'MANAGER',
            r'SOFTWARE ENGINEER|SOFTWARE DEVELOPER': 'SOFTWARE_ENGINEER',
            r'DATA SCIENTIST': 'DATA_SCIENTIST',
            r'\bENGINEER\b': 'ENGINEER',
            r'\bDEVELOPER\b': 'DEVELOPER',
            r'\bANALYST\b': 'ANALYST',
            r'BOARD MEMBER|BOARD OF DIRECTORS': 'BOARD_MEMBER',
            r'\bADVISOR\b': 'ADVISOR',
            r'\bINVESTOR\b': 'INVESTOR',
            r'\bPARTNER\b': 'PARTNER',
            r'MANAGING PARTNER': 'MANAGING_PARTNER'
        }
        
        self.subject_standardizations = {
            'COMPUTER SCIENCE AND ENGINEERING': 'COMPUTER SCIENCE',
            'COMPUTER SCIENCE & ENGINEERING': 'COMPUTER SCIENCE',
            'ELECTRICAL AND ELECTRONICS ENGINEERING': 'ELECTRICAL ENGINEERING',
            'BUSINESS ADMINISTRATION AND MANAGEMENT': 'BUSINESS ADMINISTRATION',
            'FINANCE, GENERAL': 'FINANCE',
            'MARKETING/MARKETING MANAGEMENT': 'MARKETING',
            'BUSINESS/COMMERCE, GENERAL': 'BUSINESS',
            'POLITICAL SCIENCE AND GOVERNMENT': 'POLITICAL SCIENCE'
        }
    
    def split_and_clean_safe(self, value: Any) -> Optional[list]:
        """Safely split comma and slash-separated values"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            # Convert to string and split by comma, then by slash
            items = str(value).split(',')
            all_items = []
            for item in items:
                sub_items = item.split('/')
                all_items.extend(sub_items)
            
            # Clean and deduplicate
            cleaned_items = []
            seen = set()
            for item in all_items:
                clean_item = item.strip()
                if clean_item and clean_item.lower() not in seen:
                    cleaned_items.append(clean_item)
                    seen.add(clean_item.lower())
            
            return cleaned_items if cleaned_items else np.nan
        except:
            return np.nan
    
    def standardize_degrees(self, degree_str: Any) -> list:
        """Standardize degree types using comprehensive regex patterns"""
        if pd.isna(degree_str) or degree_str == '':
            return np.nan
        
        try:
            degree_str = str(degree_str).upper().strip()
            
            # Split by common separators
            separators = [',', ';', '/', ' AND ', ' & ', ' + ']
            degrees = [degree_str]
            
            for sep in separators:
                new_degrees = []
                for degree in degrees:
                    new_degrees.extend([d.strip() for d in degree.split(sep) if d.strip()])
                degrees = new_degrees
            
            found_degrees = []
            for degree in degrees:
                degree_clean = degree.strip()
                
                # Skip very short or generic terms
                if len(degree_clean) <= 1 or degree_clean in ['DEGREE', 'UNKNOWN', 'GENERAL', '']:
                    continue
                
                matched = False
                for degree_type, pattern in self.degree_patterns.items():
                    if re.search(pattern, degree_clean):
                        if degree_type not in found_degrees:
                            found_degrees.append(degree_type)
                        matched = True
                        break
                
                # If no pattern matched and it's reasonable length, categorize as OTHER
                if not matched and len(degree_clean) > 2:
                    if 'OTHER' not in found_degrees:
                        found_degrees.append('OTHER')
            
            return found_degrees if found_degrees else ['OTHER']
            
        except:
            return ['OTHER']
    
    def consolidate_subjects(self, subject_series: pd.Series, top_n: int = 1000) -> pd.Series:
        """Keep only top N subjects by frequency, group rest as 'OTHER'"""
        if len(subject_series.dropna()) == 0:
            return subject_series
        
        # Flatten all subjects and standardize
        all_subjects = []
        for subjects in subject_series.dropna():
            if isinstance(subjects, list):
                for subject in subjects:
                    if isinstance(subject, str) and subject.strip():
                        clean_subject = subject.strip().upper()
                        
                        # Apply standardizations
                        for old_name, new_name in self.subject_standardizations.items():
                            if old_name in clean_subject:
                                clean_subject = new_name
                                break
                        
                        if clean_subject not in ['UNKNOWN', 'DEGREE', 'GENERAL', '']:
                            all_subjects.append(clean_subject)
        
        if not all_subjects:
            return subject_series
        
        # Count frequencies and find top N
        subject_counts = Counter(all_subjects)
        
        # Get top N subjects
        top_subjects = set([subject for subject, count in subject_counts.most_common(top_n)])
        
        # Calculate coverage
        top_count = sum(count for subject, count in subject_counts.most_common(top_n))
        total_count = sum(subject_counts.values())
        coverage = (top_count / total_count) * 100 if total_count > 0 else 0
        
        logger.info(f"Keeping top {top_n} subjects (covers {coverage:.1f}% of all subject mentions)")
        
        def clean_subject_list(subjects):
            try:
                if subjects is None or (isinstance(subjects, (int, float)) and np.isnan(subjects)):
                    return np.nan
                if not isinstance(subjects, list) or len(subjects) == 0:
                    return np.nan
            except (TypeError, ValueError):
                return np.nan
            
            cleaned = []
            for s in subjects:
                if isinstance(s, str) and s.strip():
                    clean_s = s.strip().upper()
                    
                    for old_name, new_name in self.subject_standardizations.items():
                        if old_name in clean_s:
                            clean_s = new_name
                            break
                    
                    if clean_s not in ['UNKNOWN', 'DEGREE', 'GENERAL', '']:
                        final_subject = clean_s if clean_s in top_subjects else 'OTHER'
                        if final_subject not in cleaned:
                            cleaned.append(final_subject)
            
            return cleaned if cleaned else np.nan
        
        return subject_series.apply(clean_subject_list)
    
    def consolidate_job_titles(self, title_series: pd.Series, top_n: int = 500) -> pd.Series:
        """Keep only top N job titles by frequency, group rest as 'OTHER'"""
        if len(title_series.dropna()) == 0:
            return title_series
        
        # Flatten all job titles and standardize
        all_titles = []
        for titles in title_series.dropna():
            if isinstance(titles, list):
                for title in titles:
                    if isinstance(title, str) and title.strip():
                        # Apply standardization patterns first
                        title_upper = title.strip().upper()
                        standardized = None
                        
                        for pattern, replacement in self.job_title_patterns.items():
                            if re.search(pattern, title_upper):
                                standardized = replacement
                                break
                        
                        # If no pattern matched, clean manually
                        if standardized is None:
                            noise_words = ['THE', 'OF', 'AND', 'FOR', 'IN', 'AT', 'TO', 'WITH']
                            words = title_upper.split()
                            cleaned_words = [w for w in words if w not in noise_words or len(words) <= 2]
                            standardized = '_'.join(cleaned_words) if cleaned_words else title_upper
                        
                        if standardized and standardized not in ['UNKNOWN', 'GENERAL', '']:
                            all_titles.append(standardized)
        
        if not all_titles:
            return title_series
        
        # Count frequencies and find top N
        title_counts = Counter(all_titles)
        
        # Get top N titles
        top_titles = set([title for title, count in title_counts.most_common(top_n)])
        
        # Calculate coverage
        top_count = sum(count for title, count in title_counts.most_common(top_n))
        total_count = sum(title_counts.values())
        coverage = (top_count / total_count) * 100 if total_count > 0 else 0
        
        logger.info(f"Keeping top {top_n} job titles (covers {coverage:.1f}% of all title mentions)")
        
        def clean_title_list(titles):
            try:
                if titles is None or (isinstance(titles, (int, float)) and np.isnan(titles)):
                    return np.nan
                if not isinstance(titles, list) or len(titles) == 0:
                    return np.nan
            except (TypeError, ValueError):
                return np.nan
            
            cleaned = []
            for title in titles:
                if isinstance(title, str) and title.strip():
                    # Apply standardization patterns
                    title_upper = title.strip().upper()
                    standardized = None
                    
                    for pattern, replacement in self.job_title_patterns.items():
                        if re.search(pattern, title_upper):
                            standardized = replacement
                            break
                    
                    # If no pattern matched, clean manually
                    if standardized is None:
                        noise_words = ['THE', 'OF', 'AND', 'FOR', 'IN', 'AT', 'TO', 'WITH']
                        words = title_upper.split()
                        cleaned_words = [w for w in words if w not in noise_words or len(words) <= 2]
                        standardized = '_'.join(cleaned_words) if cleaned_words else title_upper
                    
                    if standardized and standardized not in ['UNKNOWN', 'GENERAL', '']:
                        final_title = standardized if standardized in top_titles else 'OTHER'
                        if final_title not in cleaned:
                            cleaned.append(final_title)
            
            return cleaned if cleaned else np.nan
        
        return title_series.apply(clean_title_list)
    
    def standardize_job_titles(self, title_str: Any) -> Optional[list]:
        """Split and standardize job titles (kept for backward compatibility)"""
        if pd.isna(title_str) or title_str == '':
            return np.nan
        
        try:
            title_str = str(title_str)
            
            # Split by common separators
            separators = [' and ', ' & ', ',', '/', ' / ', ' + ']
            titles = [title_str]
            
            for sep in separators:
                new_titles = []
                for title in titles:
                    new_titles.extend([t.strip() for t in title.split(sep) if t.strip()])
                titles = new_titles
            
            # Remove duplicates
            unique_titles = []
            seen = set()
            for title in titles:
                title_clean = title.strip()
                if title_clean and title_clean.upper() not in seen:
                    unique_titles.append(title_clean)
                    seen.add(title_clean.upper())
            
            # Standardize using patterns
            standardized_titles = []
            for title in unique_titles:
                title_upper = title.upper()
                standardized = None
                
                # Try to match standardization patterns
                for pattern, replacement in self.job_title_patterns.items():
                    if re.search(pattern, title_upper):
                        standardized = replacement
                        break
                
                # If no pattern matched, clean it manually
                if standardized is None:
                    noise_words = ['THE', 'OF', 'AND', 'FOR', 'IN', 'AT', 'TO', 'WITH']
                    words = title_upper.split()
                    cleaned_words = [w for w in words if w not in noise_words or len(words) <= 2]
                    standardized = '_'.join(cleaned_words) if cleaned_words else title_upper
                
                if standardized not in standardized_titles:
                    standardized_titles.append(standardized)
            
            return standardized_titles
        except:
            return np.nan
    
    def create_quantile_bins(self, series: pd.Series, n_bins: int = 10, prefix: str = '') -> pd.Series:
        """Create quantile-based bins for continuous variables"""
        if series.isna().all() or len(series.dropna()) == 0:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        non_null_series = series.dropna()
        
        # If all values are the same, create a single bin
        if non_null_series.nunique() == 1:
            result = pd.Series([f"{prefix}_1"] * len(series), index=series.index)
            result[series.isna()] = np.nan
            return result
        
        try:
            # Remove extreme outliers (cap at 99.5th percentile)
            q99_5 = non_null_series.quantile(0.995)
            series_capped = series.clip(upper=q99_5)
            
            # Create quantile bins
            bins = pd.qcut(series_capped, q=n_bins, duplicates='drop', precision=0)
            
            # Create descriptive labels
            if hasattr(bins, 'cat') and len(bins.cat.categories) > 1:
                bin_labels = [f"{prefix}_{i+1}" for i in range(len(bins.cat.categories))]
                bins = bins.cat.rename_categories(bin_labels)
            else:
                bins = pd.Series([f"{prefix}_1"] * len(series), index=series.index)
                bins[series.isna()] = np.nan
            
            return bins
            
        except Exception as e:
            logger.warning(f"Could not create bins for {prefix}: {e}")
            # Simple fallback: binary split at median
            try:
                median_val = non_null_series.median()
                result = pd.Series(index=series.index, dtype='object')
                result[series <= median_val] = f"{prefix}_LOW"
                result[series > median_val] = f"{prefix}_HIGH"
                result[series.isna()] = np.nan
                return result
            except:
                # Ultimate fallback
                result = pd.Series([f"{prefix}_1"] * len(series), index=series.index)
                result[series.isna()] = np.nan
                return result
    
    def clean_startup_data(self, combined_events_path: str, company_base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Main cleaning function for startup dataset
        
        Args:
            combined_events_path: Path to combined events CSV
            company_base_path: Path to company base CSV
            
        Returns:
            Tuple of (cleaned_company_base, cleaned_combined_events, cleaning_summary)
        """
        logger.info("Loading data...")
        combined_events = pd.read_csv(combined_events_path)
        company_base = pd.read_csv(company_base_path)
        
        logger.info(f"Initial shapes - Combined events: {combined_events.shape}, Company base: {company_base.shape}")
        
        company_id_col = 'COMPANY_ID'
        
        # =============================================================================
        # 1. Filter companies by country code (NEW - exclude missing country codes)
        # =============================================================================
        logger.info("1. Filtering by country code...")
        
        initial_company_count = len(company_base)
        if 'country_code' in company_base.columns:
            # Remove companies with missing country codes
            company_base = company_base[company_base['country_code'].notna()].copy()
            logger.info(f"Companies filtered by country code: {initial_company_count:,} → {len(company_base):,} "
                       f"(removed {initial_company_count - len(company_base):,} companies with missing country codes)")
        else:
            logger.warning("No 'country_code' column found in company_base")
        
        # =============================================================================
        # 2. Filter companies by founding date and funding dates
        # =============================================================================
        logger.info("2. Filtering by dates...")
        
        # Convert founding date if it's string
        if 'founded_on' in company_base.columns:
            company_base['founded_on'] = pd.to_datetime(company_base['founded_on'], errors='coerce')
        
        # Filter companies founded before 2023 and after 2000
        valid_companies = company_base[
            (company_base['founded_on'] < '2023-01-01') & 
            (company_base['founded_on'] > '2000-01-01') |
            (company_base['founded_on'].isna())
        ].copy()
        
        logger.info(f"Companies filtered by founding date: {len(company_base)} → {len(valid_companies)}")
        
        # Get valid company IDs and filter events
        valid_company_ids = set(valid_companies[company_id_col].unique())
        combined_events = combined_events[combined_events[company_id_col].isin(valid_company_ids)].copy()
        
        # Convert event date and filter funding events after 2000
        combined_events['RECORD_DATE'] = pd.to_datetime(combined_events['RECORD_DATE'], errors='coerce')
        
        funding_event_types = ['INVESTMENT', 'ACQUIRED', 'ACQUISITION', 'IPO']
        funding_mask = combined_events['EVENT_TYPE'].isin(funding_event_types)
        
        combined_events = combined_events[
            (~funding_mask) | 
            (funding_mask & (combined_events['RECORD_DATE'] >= '2000-01-01'))
        ].copy()
        
        company_base = valid_companies
        
        logger.info(f"After date filtering - Combined events: {combined_events.shape}, Company base: {company_base.shape}")
        
        # =============================================================================  
        # 3. Clean comma-separated fields
        # =============================================================================
        logger.info("3. Cleaning comma-separated fields...")
        
        # Clean category fields in company_base
        if 'category_list' in company_base.columns:
            logger.info("   Cleaning category_list...")
            company_base['category_list'] = company_base['category_list'].apply(self.split_and_clean_safe)
        
        if 'category_groups_list' in company_base.columns:
            logger.info("   Cleaning category_groups_list...")
            company_base['category_groups_list'] = company_base['category_groups_list'].apply(self.split_and_clean_safe)
        
        # Clean fields in combined_events
        fields_to_split = [
            'EDU_subject',
            'EVENT_roles', 
            'INV_investor_roles',
            'INV_investor_types'
        ]
        
        for field in fields_to_split:
            if field in combined_events.columns:
                logger.info(f"   Cleaning {field}...")
                combined_events[field] = combined_events[field].apply(self.split_and_clean_safe)
        
        # =============================================================================
        # 4. Clean and standardize education degrees
        # =============================================================================
        logger.info("4. Cleaning education degrees...")
        
        if 'EDU_degree_type' in combined_events.columns:
            logger.info("   Standardizing degree types...")
            combined_events['EDU_degree_type'] = combined_events['EDU_degree_type'].apply(self.standardize_degrees)
        
        # =============================================================================  
        # 5. Consolidate education subjects (MODIFIED - Top 1000)
        # =============================================================================
        logger.info("5. Consolidating education subjects to top 1000...")
        
        if 'EDU_subject' in combined_events.columns:
            combined_events['EDU_subject'] = self.consolidate_subjects(combined_events['EDU_subject'], top_n=1000)
        
        # =============================================================================
        # 6. Clean and consolidate job titles (MODIFIED - Top 500)
        # =============================================================================
        logger.info("6. Cleaning and consolidating job titles to top 500...")
        
        if 'PEOPLE_job_title' in combined_events.columns:
            # First split and standardize job titles
            combined_events['PEOPLE_job_title'] = combined_events['PEOPLE_job_title'].apply(self.standardize_job_titles)
            # Then consolidate to top 500
            combined_events['PEOPLE_job_title'] = self.consolidate_job_titles(combined_events['PEOPLE_job_title'], top_n=500)
        
        # =============================================================================
        # 7. Convert continuous variables to quantile bins
        # =============================================================================
        logger.info("7. Converting continuous variables to quantile bins...")
        
        # Continuous variables in company_base
        company_continuous = ['num_funding_rounds', 'total_funding_usd', 'num_exits']
        for col in company_continuous:
            if col in company_base.columns:
                logger.info(f"   Binning {col}...")
                company_base[f'{col}_binned'] = self.create_quantile_bins(
                    company_base[col], prefix=col.upper()
                )
        
        # Continuous variables in combined_events  
        event_continuous = [
            'ACQ_price_usd', 'INV_fund_size_usd', 'INV_investor_count',
            'INV_investor_investment_count', 'INV_post_money_valuation_usd',
            'INV_raised_amount_usd', 'IPO_money_raised_usd', 'IPO_share_price_usd',
            'IPO_valuation_usd', 'TOTAL_EVENTS_PER_COMPANY'
        ]
        
        for col in event_continuous:
            if col in combined_events.columns:
                logger.info(f"   Binning {col}...")
                combined_events[f'{col}_binned'] = self.create_quantile_bins(
                    combined_events[col], prefix=col.upper()
                )
        
        # =============================================================================
        # 8. Final data quality checks
        # =============================================================================
        logger.info("8. Final data quality checks...")
        
        # Check for negative company ages
        if 'company_age_years' in company_base.columns:
            negative_age_mask = company_base['company_age_years'] < 0
            if negative_age_mask.any():
                logger.info(f"   Removing {negative_age_mask.sum()} companies with negative age...")
                company_base = company_base[~negative_age_mask].copy()
                
                # Update combined_events to match
                valid_company_ids = set(company_base[company_id_col].unique())
                combined_events = combined_events[
                    combined_events[company_id_col].isin(valid_company_ids)
                ].copy()
        
        # Remove any remaining invalid dates
        invalid_date_mask = combined_events['RECORD_DATE'].isna()
        if invalid_date_mask.any():
            logger.info(f"   Removing {invalid_date_mask.sum()} events with invalid dates...")
            combined_events = combined_events[~invalid_date_mask].copy()
        
        # =============================================================================
        # 9. Create summary and save results
        # =============================================================================
        logger.info("9. Creating summary and saving results...")
        
        # Create comprehensive cleaning summary
        cleaning_summary = {
            'original_company_count': initial_company_count,
            'original_event_count': len(pd.read_csv(combined_events_path)),
            'final_company_count': len(company_base),
            'final_event_count': len(combined_events),
            'companies_without_events': len(company_base) - len(set(combined_events[company_id_col].unique())),
            'event_type_distribution': combined_events['EVENT_TYPE'].value_counts().to_dict() if 'EVENT_TYPE' in combined_events.columns else {},
            'cleaning_steps_completed': [
                'Country code filtering (exclude missing country codes)',
                'Date filtering (companies before 2023, funding after 2000)',
                'Comma-separated fields splitting',
                'Education degree standardization with regex',
                'Education subject consolidation (top 1000)',
                'Job title standardization and consolidation (top 500)',
                'Continuous variable binning',
                'Data quality checks and validation'
            ],
            'cleaning_date': datetime.now().isoformat()
        }
        
        cleaning_summary['companies_without_events_pct'] = (
            cleaning_summary['companies_without_events'] / cleaning_summary['final_company_count'] * 100
        )
        
        # Save results
        company_base.to_csv(f"{self.output_dir}/company_base_cleaned.csv", index=False)
        combined_events.to_csv(f"{self.output_dir}/combined_events_cleaned.csv", index=False)
        
        with open(f"{self.output_dir}/company_base_cleaned.pkl", 'wb') as f:
            pickle.dump(company_base, f)
        
        with open(f"{self.output_dir}/combined_events_cleaned.pkl", 'wb') as f:
            pickle.dump(combined_events, f)
        
        with open(f"{self.output_dir}/cleaning_summary.pkl", 'wb') as f:
            pickle.dump(cleaning_summary, f)
        
        logger.info("✅ Data cleaning completed!")
        logger.info(f"📁 Files saved to: {self.output_dir}/")
        
        return company_base, combined_events, cleaning_summary


def generate_summary_statistics(company_base: pd.DataFrame, combined_events: pd.DataFrame, cleaning_summary: Optional[Dict] = None):
    """
    Generate comprehensive summary statistics for cleaned startup data
    """
    print("="*80)
    print("📊 COMPREHENSIVE DATA SUMMARY STATISTICS")
    print("="*80)
    
    # =============================================================================
    # 1. Overall Data Shape & Quality
    # =============================================================================
    print("\n1. OVERALL DATA SHAPE & QUALITY")
    print("-" * 40)
    print(f"Companies: {len(company_base):,}")
    print(f"Events: {len(combined_events):,}")
    
    # Companies with/without events
    companies_with_events = set(combined_events['COMPANY_ID'].unique())
    companies_without_events = len(company_base) - len(companies_with_events)
    print(f"Companies with events: {len(companies_with_events):,} ({len(companies_with_events)/len(company_base)*100:.1f}%)")
    print(f"Companies without events: {companies_without_events:,} ({companies_without_events/len(company_base)*100:.1f}%)")
    
    # Date ranges
    if 'founded_on' in company_base.columns:
        print(f"\nFounding date range: {company_base['founded_on'].min().strftime('%Y-%m-%d')} to {company_base['founded_on'].max().strftime('%Y-%m-%d')}")
    
    if 'RECORD_DATE' in combined_events.columns:
        print(f"Event date range: {combined_events['RECORD_DATE'].min().strftime('%Y-%m-%d')} to {combined_events['RECORD_DATE'].max().strftime('%Y-%m-%d')}")
    
    # =============================================================================
    # 2. Company Base Statistics
    # =============================================================================
    print("\n\n2. COMPANY BASE STATISTICS")
    print("-" * 40)
    
    # Missing values
    print("Missing values in key fields:")
    key_company_fields = ['founded_on', 'category_list', 'category_groups_list', 'country_code', 'status']
    for field in key_company_fields:
        if field in company_base.columns:
            missing_pct = (company_base[field].isna().sum() / len(company_base)) * 100
            print(f"  {field}: {missing_pct:.1f}%")
    
    # Category distribution
    if 'category_list' in company_base.columns:
        print("\nTop 10 Categories (first category only):")
        # Get first category from each list
        first_categories = []
        for cat_list in company_base['category_list'].dropna():
            if isinstance(cat_list, list) and len(cat_list) > 0:
                first_categories.append(cat_list[0])
        
        if first_categories:
            cat_counts = pd.Series(first_categories).value_counts()
            for cat, count in cat_counts.head(10).items():
                print(f"  {cat}: {count:,} ({count/len(company_base)*100:.1f}%)")
    
    # Status distribution
    if 'status' in company_base.columns:
        print("\nCompany Status Distribution:")
        status_counts = company_base['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count:,} ({count/len(company_base)*100:.1f}%)")
    
    # Geographic distribution
    if 'country_code' in company_base.columns:
        print("\nTop 10 Countries:")
        country_counts = company_base['country_code'].value_counts()
        for country, count in country_counts.head(10).items():
            print(f"  {country}: {count:,} ({count/len(company_base)*100:.1f}%)")
    
    # =============================================================================
    # 3. Events Statistics
    # =============================================================================
    print("\n\n3. EVENTS STATISTICS")
    print("-" * 40)
    
    # Event type distribution
    if 'EVENT_TYPE' in combined_events.columns:
        print("Event Type Distribution:")
        event_counts = combined_events['EVENT_TYPE'].value_counts()
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count:,} ({count/len(combined_events)*100:.1f}%)")
    
    # Events per company
    events_per_company = combined_events.groupby('COMPANY_ID').size()
    print(f"\nEvents per Company Statistics:")
    print(f"  Mean: {events_per_company.mean():.1f}")
    print(f"  Median: {events_per_company.median():.1f}")
    print(f"  Min: {events_per_company.min()}")
    print(f"  Max: {events_per_company.max()}")
    print(f"  75th percentile: {events_per_company.quantile(0.75):.1f}")
    print(f"  95th percentile: {events_per_company.quantile(0.95):.1f}")
    
    # =============================================================================
    # 4. Cleaned Fields Quality Check (ENHANCED)
    # =============================================================================
    print("\n\n4. CLEANED FIELDS QUALITY CHECK")
    print("-" * 40)
    
    # Check list fields
    list_fields = {
        'EVENT_roles': 'Event Roles',
        'PEOPLE_job_title': 'Job Titles', 
        'EDU_subject': 'Education Subjects',
        'EDU_degree_type': 'Degree Types',
        'INV_investor_roles': 'Investor Roles',
        'INV_investor_types': 'Investor Types'
    }
    
    for field, display_name in list_fields.items():
        if field in combined_events.columns:
            # Count unique values across all lists
            all_items = []
            non_null_count = 0
            for item_list in combined_events[field].dropna():
                if isinstance(item_list, list):
                    all_items.extend(item_list)
                    non_null_count += 1
            
            unique_count = len(set(all_items))
            total_items = len(all_items)
            avg_items_per_record = total_items / non_null_count if non_null_count > 0 else 0
            
            print(f"\n{display_name}:")
            print(f"  Records with data: {non_null_count:,}")
            print(f"  Total items: {total_items:,}")
            print(f"  Unique items: {unique_count:,}")
            print(f"  Avg items per record: {avg_items_per_record:.1f}")
            
            # Show top 5 most common items
            if all_items:
                item_counts = pd.Series(all_items).value_counts()
                print(f"  Top 5 most common:")
                for item, count in item_counts.head(5).items():
                    print(f"    {item}: {count:,}")
                
                # Special reporting for job titles and education subjects
                if field in ['PEOPLE_job_title', 'EDU_subject']:
                    other_count = item_counts.get('OTHER', 0)
                    if other_count > 0:
                        other_pct = (other_count / total_items) * 100
                        print(f"  'OTHER' category: {other_count:,} ({other_pct:.1f}%)")
                        top_coverage = ((total_items - other_count) / total_items) * 100
                        print(f"  Top categories coverage: {top_coverage:.1f}%")
    
    # =============================================================================
    # 5. Data Quality Indicators
    # =============================================================================
    print("\n\n5. DATA QUALITY INDICATORS")
    print("-" * 40)
    
    # Overall completeness
    company_completeness = (1 - company_base.isna().sum().sum() / (len(company_base) * len(company_base.columns))) * 100
    events_completeness = (1 - combined_events.isna().sum().sum() / (len(combined_events) * len(combined_events.columns))) * 100
    
    print(f"Overall data completeness:")
    print(f"  Company base: {company_completeness:.1f}%")
    print(f"  Events: {events_completeness:.1f}%")
    
    # Check for potential data issues
    issues = []
    
    # Check for extremely high values in binned columns
    for col in combined_events.columns:
        if 'binned' in col:
            if combined_events[col].notna().sum() == 0:
                issues.append(f"Column {col} has no data after binning")
    
    # Check for companies with very old founding dates
    if 'founded_on' in company_base.columns:
        very_old = company_base[company_base['founded_on'] < '1950-01-01']
        if len(very_old) > 0:
            issues.append(f"{len(very_old)} companies founded before 1950")
    
    # Check for future events
    if 'RECORD_DATE' in combined_events.columns:
        future_events = combined_events[combined_events['RECORD_DATE'] > pd.Timestamp.now()]
        if len(future_events) > 0:
            issues.append(f"{len(future_events)} events in the future")
    
    if issues:
        print(f"\nPotential data issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"\n✅ No major data quality issues detected")
    
    # =============================================================================
    # 6. Memory Usage
    # =============================================================================
    print("\n\n6. MEMORY USAGE")
    print("-" * 40)
    
    company_memory = company_base.memory_usage(deep=True).sum() / 1024**2
    events_memory = combined_events.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Memory usage:")
    print(f"  Company base: {company_memory:.1f} MB")
    print(f"  Events: {events_memory:.1f} MB")
    print(f"  Total: {company_memory + events_memory:.1f} MB")
    
    # =============================================================================
    # 7. Cleaning Summary (if provided)
    # =============================================================================
    if cleaning_summary:
        print("\n\n7. CLEANING SUMMARY")
        print("-" * 40)
        print(f"Original companies: {cleaning_summary['original_company_count']:,}")
        print(f"Final companies: {cleaning_summary['final_company_count']:,}")
        print(f"Companies removed: {cleaning_summary['original_company_count'] - cleaning_summary['final_company_count']:,}")
        print(f"Original events: {cleaning_summary['original_event_count']:,}")
        print(f"Final events: {cleaning_summary['final_event_count']:,}")
        print(f"Events removed: {cleaning_summary['original_event_count'] - cleaning_summary['final_event_count']:,}")
        
        print(f"\nCleaning steps completed:")
        for step in cleaning_summary['cleaning_steps_completed']:
            print(f"  ✅ {step}")
    
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS COMPLETE")
    print("="*80)


# Convenience function for backward compatibility
def clean_startup_data(combined_events_path: str, company_base_path: str, output_dir: str = "data/cleaned/cleaned_startup") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to clean startup data using the StartupDataCleaner class
    
    Args:
        combined_events_path: Path to combined events CSV
        company_base_path: Path to company base CSV  
        output_dir: Directory to save cleaned data
        
    Returns:
        Tuple of (cleaned_company_base, cleaned_combined_events, cleaning_summary)
    """
    cleaner = StartupDataCleaner(output_dir=output_dir)
    return cleaner.clean_startup_data(combined_events_path, company_base_path)
