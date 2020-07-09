import xlwt
import xlrd

outputFileLocation = '../output/charts_spreadsheet.xls'

def readColumn(sheet, column, fromRow, toRow):
    columnContents = []
    for row in range(fromRow, toRow + 1):
        columnContents.append(sheet.cell(row, column).value)
    return columnContents

def writeColumn(sheet, columnContents, column, fromRow, toRow):
    for row in range(fromRow, toRow + 1):
        sheet.write(row, column, columnContents[row - fromRow])

print('Reading master spreadsheet..')

master_spreadsheet = xlrd.open_workbook('../output/master_spreadsheet.xls')
book = xlwt.Workbook(encoding="utf-8")

print('Generating matching performance spreadsheet..')

matchingPerformanceSheet = book.add_sheet("Figure 10 Matching Performance")

# Write headers
matchingPerformanceHeaders = \
    ['Experiment Index',
     'RICI with 1 uncluttered object', 'RICI with 4 added clutter objects', 'RICI with 9 added clutter objects',
     'SI with 1 uncluttered object',   'SI with 4 added clutter objects',   'SI with 9 added clutter objects',
     '3DSC with 1 uncluttered object', '3DSC with 4 added clutter objects', '3DSC with 9 added clutter objects']

for i in range(0, len(matchingPerformanceHeaders)):
    matchingPerformanceSheet.write(0, i, matchingPerformanceHeaders[i])

writeColumn(matchingPerformanceSheet, [x for x in range(1, 1501)], 0, 1, 1500)

# Read data from master spreadsheet
rici_matching_performance_1_object = readColumn(master_spreadsheet.sheet_by_name('Rank 0 RICI results'), 6, 1, 1500)
rici_matching_performance_5_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 RICI results'), 7, 1, 1500)
rici_matching_performance_10_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 RICI results'), 8, 1, 1500)

si_matching_performance_1_object = readColumn(master_spreadsheet.sheet_by_name('Rank 0 SI results'), 9, 1, 1500)
si_matching_performance_5_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 SI results'), 10, 1, 1500)
si_matching_performance_10_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 SI results'), 11, 1, 1500)

sc_matching_performance_1_object = readColumn(master_spreadsheet.sheet_by_name('Rank 0 3DSC results'), 3, 1, 1500)
sc_matching_performance_5_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 3DSC results'), 4, 1, 1500)
sc_matching_performance_10_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 3DSC results'), 5, 1, 1500)

# Sort each sequence individually for better readability
rici_matching_performance_1_object.sort()
rici_matching_performance_5_objects.sort()
rici_matching_performance_10_objects.sort()

si_matching_performance_1_object.sort()
si_matching_performance_5_objects.sort()
si_matching_performance_10_objects.sort()

sc_matching_performance_1_object.sort()
sc_matching_performance_5_objects.sort()
sc_matching_performance_10_objects.sort()

# Write data to spreadsheet
writeColumn(matchingPerformanceSheet, rici_matching_performance_1_object, 1, 1, 1500)
writeColumn(matchingPerformanceSheet, rici_matching_performance_5_objects, 2, 1, 1500)
writeColumn(matchingPerformanceSheet, rici_matching_performance_10_objects, 3, 1, 1500)

writeColumn(matchingPerformanceSheet, si_matching_performance_1_object, 4, 1, 1500)
writeColumn(matchingPerformanceSheet, si_matching_performance_5_objects, 5, 1, 1500)
writeColumn(matchingPerformanceSheet, si_matching_performance_10_objects, 6, 1, 1500)

writeColumn(matchingPerformanceSheet, sc_matching_performance_1_object, 7, 1, 1500)
writeColumn(matchingPerformanceSheet, sc_matching_performance_5_objects, 8, 1, 1500)
writeColumn(matchingPerformanceSheet, sc_matching_performance_10_objects, 9, 1, 1500)

print('Generating spin image support angle spreadsheet..')

supportAngleSheet = book.add_sheet("Figure 11 Support Angle")

# Write headers
supportAngleHeaders = \
    ['Experiment Index',
     'SI, no support angle, 1 uncluttered object',
     'SI, no support angle, 4 added clutter objects',
     'SI, no support angle, 9 added clutter objects',
     'SI, 60° support angle, 1 uncluttered object',
     'SI, 60° support angle, 4 added clutter objects',
     'SI, 60° support angle, 9 added clutter objects']

for i in range(0, len(supportAngleHeaders)):
    supportAngleSheet.write(0, i, supportAngleHeaders[i])

writeColumn(supportAngleSheet, [x for x in range(1, 1501)], 0, 1, 1500)

# Can reuse the columns from before as they are the same
writeColumn(supportAngleSheet, si_matching_performance_1_object, 1, 1, 1500)
writeColumn(supportAngleSheet, si_matching_performance_5_objects, 2, 1, 1500)
writeColumn(supportAngleSheet, si_matching_performance_10_objects, 3, 1, 1500)

# 60 degree matching results are new, though
si_60_matching_performance_1_object = readColumn(master_spreadsheet.sheet_by_name('Rank 0 SI results'), 12, 1, 1500)
si_60_matching_performance_5_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 SI results'), 13, 1, 1500)
si_60_matching_performance_10_objects = readColumn(master_spreadsheet.sheet_by_name('Rank 0 SI results'), 14, 1, 1500)

si_60_matching_performance_1_object.sort()
si_60_matching_performance_5_objects.sort()
si_60_matching_performance_10_objects.sort()

writeColumn(supportAngleSheet, si_60_matching_performance_1_object, 4, 1, 1500)
writeColumn(supportAngleSheet, si_60_matching_performance_5_objects, 5, 1, 1500)
writeColumn(supportAngleSheet, si_60_matching_performance_10_objects, 6, 1, 1500)

print('Generating descriptor generation rate spreadsheet..')

generationRateSheet = book.add_sheet("Figure 13 Generation Rates")

# Write headers
generationRateHeaders = \
    ['Triangle Count',
     'RICI',
     'SI',
     '3DSC']

for i in range(0, len(generationRateHeaders)):
    generationRateSheet.write(0, i, generationRateHeaders[i])

rici_generation_times = readColumn(master_spreadsheet.sheet_by_name('RICI Generation Times'), 2, 1, 1500)
si_generation_times = readColumn(master_spreadsheet.sheet_by_name('SI Generation Times'), 10, 1, 1500)
sc_generation_times = readColumn(master_spreadsheet.sheet_by_name('3DSC Generation Times'), 4, 1, 1500)

scene_image_counts_5_objects = readColumn(master_spreadsheet.sheet_by_name('Total Image Count'), 1, 1, 1500)
triangle_counts_5_objects = readColumn(master_spreadsheet.sheet_by_name('Total Triangle Count'), 1, 1, 1500)

# Generation rate is image count / time taken -> images / second
rici_generation_rates = [imageCount / rici_generation_times[index] for index, imageCount in enumerate(scene_image_counts_5_objects)]
si_generation_rates = [imageCount / si_generation_times[index] for index, imageCount in enumerate(scene_image_counts_5_objects)]
sc_generation_rates = [imageCount / sc_generation_times[index] for index, imageCount in enumerate(scene_image_counts_5_objects)]

writeColumn(generationRateSheet, triangle_counts_5_objects, 0, 1, 1500)
writeColumn(generationRateSheet, rici_generation_rates, 1, 1, 1500)
writeColumn(generationRateSheet, si_generation_rates, 2, 1, 1500)
writeColumn(generationRateSheet, sc_generation_rates, 3, 1, 1500)

print('Generating descriptor comparison rate spreadsheet..')

comparisonRateSheet = book.add_sheet("Figure 14 Comparison Rates")

# Write headers
comparisonRateHeaders = \
    ['Experiment Index',
     'RICI without early exit',
     'RICI with early exit',
     'SI',
     '3DSC']

for i in range(0, len(comparisonRateHeaders)):
    comparisonRateSheet.write(0, i, comparisonRateHeaders[i])

writeColumn(comparisonRateSheet, [x for x in range(1, 1501)], 0, 1, 1500)

# Compute total number of comparisons done
# Equal to number of needle images times the number of haystack images
reference_image_counts_5_objects = readColumn(master_spreadsheet.sheet_by_name('Reference Image Count'), 1, 1, 1500)
comparison_counts_5_objects = [reference_image_counts_5_objects[index] * scene_image_counts_5_objects[index]
                               for index in range(0, len(reference_image_counts_5_objects))]

rici_noearlyexit_comparison_times = readColumn(master_spreadsheet.sheet_by_name('RICI Comparison Times'), 1, 1, 1500)
rici_earlyexit_comparison_times = readColumn(master_spreadsheet.sheet_by_name('RICI Comparison Times'), 2, 1, 1500)
si_comparison_times = readColumn(master_spreadsheet.sheet_by_name('SI Comparison Times'), 10, 1, 1500)
sc_comparison_times = readColumn(master_spreadsheet.sheet_by_name('3DSC Comparison Times'), 4, 1, 1500)

# Comparison rate is comparisons / time taken
rici_noearlyexit_comparison_rates = [comparison_counts_5_objects[index] / rici_noearlyexit_comparison_times[index]
                                     for index in range(0, len(comparison_counts_5_objects))]
rici_earlyexit_comparison_rates = [comparison_counts_5_objects[index] / rici_earlyexit_comparison_times[index]
                                   for index in range(0, len(comparison_counts_5_objects))]
si_comparison_rates = [comparison_counts_5_objects[index] / si_comparison_times[index]
                       for index in range(0, len(comparison_counts_5_objects))]
sc_comparison_rates = [comparison_counts_5_objects[index] / sc_comparison_times[index]
                       for index in range(0, len(comparison_counts_5_objects))]

# Next, we sort the comparison rates for readability
rici_noearlyexit_comparison_rates.sort()
rici_earlyexit_comparison_rates.sort()
si_comparison_rates.sort()
sc_comparison_rates.sort()

writeColumn(comparisonRateSheet, rici_noearlyexit_comparison_rates, 1, 1, 1500)
writeColumn(comparisonRateSheet, rici_earlyexit_comparison_rates, 2, 1, 1500)
writeColumn(comparisonRateSheet, si_comparison_rates, 3, 1, 1500)
writeColumn(comparisonRateSheet, sc_comparison_rates, 4, 1, 1500)

print('Writing output file..')

book.save(outputFileLocation)

print('Done. Results were written to: ' + outputFileLocation)